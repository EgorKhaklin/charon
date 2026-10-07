"""Comparison figures for the README, drawn from the saved tables in results/ (no rerun).

    python -m charon.figures
"""

import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .experiments import COLORS, FIG, GRID, INK, INK2, RES  # noqa: E402

# One color per method everywhere. Baselines are neutral, methods take the categorical slots.
STYLE = {
    "L1": dict(color=INK2, lw=2),
    "reweighted L1": dict(color="#a8a6a1", lw=2, ls="--"),
    "iterative support detection": dict(color=COLORS[1], lw=2),
    "Pandora": dict(color=COLORS[0], lw=2.5),
    "log wormhole": dict(color=COLORS[2], lw=2),
    "u² − v²": dict(color=COLORS[3], lw=2),
}


def tables(path):
    """Every markdown table in a results file, as (header, rows of cells)."""
    out, cur = [], None
    for line in (RES / path).read_text().splitlines():
        if line.startswith("|") and not line.startswith("|---"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if cur is None:
                cur = (cells, [])
            else:
                extra = len(cells) - len(cur[0])  # a "|" inside a name, like w|w|, in column 1
                cur[1].append(["|".join(cells[:extra + 1])] + cells[extra + 1:] if extra > 0
                              else cells)
        elif cur is not None and not line.startswith("|---"):
            out.append(cur)
            cur = None
    if cur is not None:
        out.append(cur)
    return out


def num(cell):
    return float(re.match(r"[-\d.]+", cell).group())


def pandora_vs_l1():
    head, rows = tables("pandora.txt")[0]
    k = [num(r[0]) for r in rows]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    offsets = {"L1": -10, "reweighted L1": 8, "iterative support detection": 0, "Pandora": 0}
    for col, name in [(1, "L1"), (2, "reweighted L1"), (3, "iterative support detection"),
                      (4, "Pandora")]:
        y = [num(r[col]) for r in rows]
        st = STYLE[name]
        ax.plot(k, y, color=st["color"], lw=st["lw"], ls=st.get("ls", "-"), marker="o", ms=5)
        for kk, yy in zip(k, y):
            if name in ("Pandora", "L1"):
                ax.annotate(f"{yy:.0f}", (kk, yy), xytext=(0, 7 if name == "Pandora" else -12),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.annotate(name, (k[1], y[1]), xytext=(8, offsets[name]), textcoords="offset points",
                    va="center", fontsize=9, color=INK)
    ax.set_xlim(7, 18)
    ax.set_ylim(-3, 103)
    ax.set_xticks(k)
    ax.set_xlabel("nonzeros in the true signal (40 equations, 200 unknowns)")
    ax.set_ylabel("exact recoveries out of 100")
    ax.set_title("Pandora recovers what L1 and the strongest baseline cannot", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "pandora_vs_l1.png")
    plt.close(fig)


def ceiling():
    head, rows = [t for t in tables("frontier.txt") if "L1 exact" in t[0]][0]
    rows = [r for r in rows if r[0] == "uniform 1 to 2"]
    ks = sorted({num(r[1]) for r in rows})
    l1 = [num(next(r for r in rows if num(r[1]) == k)[4]) for k in ks]
    worm = {name: [num(next(r for r in rows if num(r[1]) == k and r[2].startswith(prefix))[3])
                   for k in ks]
            for name, prefix in [("log wormhole", "log wormhole"), ("u² − v²", "u^2 - v^2")]}
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(ks))
    w = 0.26
    for i, (name, y) in enumerate([("L1", l1)] + list(worm.items())):
        bars = ax.bar(x + (i - 1) * w, y, width=w - 0.03, color=STYLE[name]["color"], label=name)
        for b, v in zip(bars, y):
            ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
    ax.set_axisbelow(True)
    ax.grid(axis="x", visible=False)
    ax.set_xticks(x, [f"{k:.0f}" for k in ks])
    ax.set_xlabel("nonzeros in the true signal")
    ax.set_ylabel("exact recoveries out of 100")
    ax.set_ylim(0, 110)
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title("The ceiling: no fixed wormhole recovers a signal L1 misses (0 of 1600)",
                 loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "ceiling.png")
    plt.close(fig)


def quantum():
    t13, t14 = [t for t in tables("quantum.txt")]
    head, rows = t13
    ms = sorted({num(r[0]) for r in rows})
    names = [("positivity, convex", INK2, "convex (positivity)"),
             ("Born rho = AA†, start 0.001", COLORS[0], "Born wormhole ρ = AA†"),
             ("Gibbs rho = exp(H)", COLORS[1], "Gibbs ρ = exp(H)"),
             ("least squares (min norm)", "#a8a6a1", "least squares")]
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.2))
    for key, color, label in names:
        y = [num(next(r for r in rows if num(r[0]) == m and r[1] == key)[2]) for m in ms]
        a.plot(ms, y, "-o", color=color, lw=2, ms=5, label=label)
    a.set_xlabel("Pauli measurements (of 255)")
    a.set_ylabel("fidelity with the true pure state")
    a.set_ylim(0, 1.05)
    a.legend(fontsize=8, loc="lower right")
    a.set_title("No noise: with few measurements, the wormhole picks", loc="left", fontsize=10)
    head, rows = t14
    groups = [r for r in rows if r[1] == "1000" and r[2] == "160"] + \
             [r for r in rows if r[1] == "100" and r[2] == "160"]
    labels = [f"{'pure' if r[0] == '1' else 'mixed'}, {r[1]} shots" for r in groups]
    x = np.arange(len(groups))
    w = 0.38
    for i, (col, color, label) in enumerate([(4, INK2, "convex (positivity)"),
                                             (6, COLORS[0], "Born wormhole")]):
        vals = [num(r[col]) for r in groups]
        bars = b.bar(x + (i - 0.5) * w, vals, width=w - 0.03, color=color, label=label)
        for bar, v in zip(bars, vals):
            b.annotate(f"{v:.2f}", (bar.get_x() + bar.get_width() / 2, v), xytext=(0, 2),
                       textcoords="offset points", ha="center", fontsize=8, color=INK2)
    b.set_axisbelow(True)
    b.grid(axis="x", visible=False)
    b.set_xticks(x, labels, fontsize=8)
    b.set_ylim(0, 1.1)
    b.set_ylabel("fidelity (160 measurements)")
    b.legend(fontsize=8, loc="upper right")
    b.set_title("Shot noise: a prior that helps pure states", loc="left", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG / "quantum.png")
    plt.close(fig)


def atlas():
    head, rows = tables("atlas.txt")[0]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), sharey=True)
    for ax, sigma in zip(axes, ("0.01", "0.05")):
        sel = [r for r in rows if r[0] == sigma]
        ks = [r[1] for r in sel]
        x = np.arange(len(ks))
        w = 0.27
        for i, (col, color, label) in enumerate([(2, "#a8a6a1", "lasso (cross-validated)"),
                                                 (3, INK2, "lasso, then least squares"),
                                                 (4, COLORS[0], "Atlas")]):
            vals = [num(r[col]) for r in sel]
            bars = ax.bar(x + (i - 1) * w, vals, width=w - 0.03, color=color, label=label)
            for b, v in zip(bars, vals):
                ax.annotate(f"{v:.0f}", (b.get_x() + b.get_width() / 2, v), xytext=(0, 2),
                            textcoords="offset points", ha="center", fontsize=8, color=INK2)
        ax.set_axisbelow(True)
        ax.grid(axis="x", visible=False)
        ax.set_xticks(x, [f"{k} nonzeros" for k in ks])
        ax.set_title(f"noise sigma = {sigma}", loc="left", fontsize=10)
    axes[0].set_ylabel("recovered within 5%, out of 40")
    axes[0].set_ylim(0, 44)
    axes[1].legend(fontsize=8, loc="upper right")
    fig.suptitle("Atlas: Pandora's idea survives noisy measurements", x=0.01, ha="left",
                 color=INK, weight="bold")
    fig.tight_layout()
    fig.savefig(FIG / "atlas.png")
    plt.close(fig)


def main():
    plt.rcParams["grid.color"] = GRID
    atlas()
    pandora_vs_l1()
    ceiling()
    quantum()


if __name__ == "__main__":
    main()
