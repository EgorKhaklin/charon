# charon

### What a parameter transform does to gradient descent.

[![Tests](https://github.com/EgorKhaklin/charon/actions/workflows/tests.yml/badge.svg)](https://github.com/EgorKhaklin/charon/actions/workflows/tests.yml)
[![License](https://img.shields.io/badge/license-MIT-2a78d6?style=flat-square)](LICENSE)

In myth, Charon ferries things to the other side. Here the parameter makes the crossing.
The optimizer moves `w`, but the model sees `T(w)`:

```
ordinary:   w ───────────▶ y_hat = w·x + b      ──▶ loss      dL/dw = mean(err·x)
charon:     w ──▶ T(w) ──▶ y_hat = T(w)·x + b   ──▶ loss      dL/dw = mean(err·x)·T'(w)
```

The idea started from E = mc². Can a physics-inspired map from the parameter you train to the
weight the model uses change how learning goes? This repository asks that question
of linear regression and checks every answer it gives.

## The answer, in one line

**A transform changes the route the optimizer takes. It never changes where the route ends.**
To first order, one gradient step on `w` moves the effective weight `v = T(w)` by

```
Δv ≈ −lr · T'(w)² · dL/dv
```

So every transform is a learning rate that depends on position, `lr·T'(w)²`. The only other
thing it adds is a range: `w²` and `exp(w)` can't go negative, and `c·tanh(w/c)` can't
reach `c`. A test checks this identity for every transform (`test_one_step_is_preconditioned_gd_on_v`).

And E = mc² itself? As a transform, `T(w) = c²·w` is plain linear regression with the learning
rate multiplied by `c⁴ ≈ 8×10³³`. It only converges for `lr < 7×10⁻³⁶` (E5). The constant
carries no information. The shape of `T` is what matters.

## The transforms

| T(w) | T'(w) | what it does |
|---|---|---|
| `w` | 1 | ordinary regression, the baseline |
| `w²` | 2w | range [0, ∞). Two mirror minima, and a dead point at w = 0 where training never starts |
| `w³` | 3w² | full range, but T'(0) = 0, so steps shrink to nothing near zero and blow up far away |
| `exp(w)` | exp(w) | range (0, ∞). Multiplicative updates, the same family as exponentiated gradient |
| `c·tanh(w/c)` | sech²(w/c) | special relativity's rapidity: velocity = c·tanh(rapidity). A speed limit, \|T(w)\| < c |

## What the experiments found

Data: `y = 3x + 2 + N(0, 2²)`, 100 points on [−10, 10]. Least squares gives slope 2.958 and loss 1.1645.
Every number below comes from `python -m charon.experiments` (full output in
[results/run.txt](results/run.txt)).

**E1. Same data, five different landscapes.** Ordinary regression has one bowl. `w²` has two
minima and a saddle point. `w³` and `exp` squeeze the minimum into a narrow valley. The
rapidity transform stretches it out.

![landscape](figures/landscape.png)

**E2. Tune the learning rate for each transform before you compare them.** The original sketch
gave every model `lr = 0.01`. At that rate `w²`, `w³` and `exp` blow up near the answer, so a
shared learning rate only shows which models it happens to suit. The curvature at the optimum
is `T'(w*)²·E[x²]`, so the largest stable rate is `2 / (T'(w*)²·E[x²])`. That prediction
matches the measurements:

| transform | best lr | steps at best lr | predicted max lr | largest lr that converged |
|---|---|---|---|---|
| identity | 0.032 | 4 | 0.0588 | 0.0501 |
| w² | 0.0025 | 6 | 0.00497 | 0.00398 |
| w³ | 0.00079 | 11 | 0.00154 | 0.00126 |
| exp(w) | 0.0032 | 6 | 0.00672 | 0.00631 |
| c·tanh(w/c), c=5 | 0.063 | 8 | 0.139 | 0.126 |

Ordinary regression wins this one, and nothing can beat it. Its loss is an exact quadratic,
so with `lr = 1/E[x²]` it lands on the answer in a single step (tested). Every other transform
bends that bowl.

![learning rate](figures/learning_rate.png)

**E3. Where training starts decides whether it works.** 201 starting points from −5 to 5, each
transform at its best learning rate. The flipped run uses the true slope −3:

| true slope | transform | converged | diverged | stalled | can't reach it |
|---|---|---|---|---|---|
| +2.96 | identity | 201 | 0 | 0 | 0 |
| +2.96 | w² | 152 | 48 | 1 | 0 |
| +2.96 | w³ | 49 | 112 | 40 | 0 |
| +2.96 | exp(w) | 149 | 0 | 52 | 0 |
| +2.96 | c·tanh(w/c) | 201 | 0 | 0 | 0 |
| −2.96 | w² | 0 | 82 | 0 | 119 |
| −2.96 | exp(w) | 0 | 0 | 0 | 201 |

The curvature changes from place to place, so one learning rate can't work from every start.
`w³` diverges from far away and stalls near zero. `exp` stalls from negative starts,
where exp(w) is tiny. `w²` never moves from exactly zero. The rapidity transform keeps
`T'(w) ≤ 1`, so its effective step `lr·T'(w)²` never exceeds the learning rate. It is the one
transform here as robust as ordinary regression.

![init map](figures/init_map.png)

**E4. Noise can't tell the transforms apart.** At noise levels from 0.1 to 10, every
transform reaches the least-squares loss to within float precision. Noise changes the answer, and
every transform finds the same changed answer. Raw final loss mostly measures the noise, so
the table reports the excess over least squares instead.

**E6. Where a transform does change the answer.** With more unknowns than equations,
many weights fit the data exactly, and the route decides which one you get. Take 40
equations, 200 unknowns and a true weight with 5 nonzeros. Plain gradient descent from zero
lands on the minimum-norm fit, which is spread out and wrong. Writing `w = u² − v²` and
starting small lands on the sparse truth:

| parameterization | steps to fit (residual 1e-8) | error vs true w | true nonzeros found |
|---|---|---|---|
| `w` | 500 | 0.913 | 2/5 |
| `u² − v²`, init 0.1 | 5,000 | 0.649 | 3/5 |
| `u² − v²`, init 0.01 | 36,000 | 0.288 | 5/5 |
| `u² − v²`, init 0.0001 | over 400,000 (residual 4e-6) | 0.028 | 5/5 |

The smaller the starting size, the sparser the answer and the slower the training. The
smallest start costs about 1000x the steps of plain descent, and its last digits of fit arrive
very slowly, because the coordinates still carrying the residual are tiny and so are their
steps. Even this run is not exactly sparse: a few wrong coordinates reach about 0.03. Here the
minimum-L1 fit equals the true weight exactly (checked once with a linear program), and the
squared parameterization only approaches it as the start shrinks toward zero (a start of 1e-8
gave error 0.004 in a side run). This is the known implicit bias of a squared
reparameterization. It is the setting where the "wormhole" idea has real work to do.

![sparse](figures/sparse.png)

## So which of the four outcomes?

The original sketch listed four possible conclusions. For one-dimensional linear regression
the answer is **3 and 4**. The transform changes optimization but brings no overall
advantage, and it behaves exactly like known techniques: a preconditioner, a positivity
constraint (`exp`), a bounded weight (`tanh`), a squared factorization (`w²`). In the
overparameterized case (E6), outcome 1 holds in a precise sense. The transform picks a better
answer among many exact fits. It does not speed up the fit.

Related work, for anyone going further: natural gradient (Amari, 1998) for transforms viewed as
preconditioning; exponentiated gradient (Kivinen and Warmuth, 1997) for `exp`; weight
normalization (Salimans and Kingma, 2016) for reparameterization in deep networks; and
Woodworth et al., "Kernel and Rich Regimes in Overparametrized Models" (COLT 2020), for E6.

## Run it

```bash
python -m venv .venv && .venv/bin/pip install -e '.[test]'
.venv/bin/python -m pytest -q              # 24 tests
.venv/bin/python -m charon.experiments     # ~12 s; writes figures/ and results/
```

To add a transform, define `T`, `dT` and `inverse` in [charon/transforms.py](charon/transforms.py)
and add it to `DEFAULT`. The gradient test (a finite-difference check) and the preconditioning
test cover it automatically.

## Limits

This is one-dimensional regression on synthetic data, plus one small sparse-recovery problem.
Nothing here is a claim about deep networks, and the "best" learning rates come from a grid,
not an optimizer. The stability limit is a local prediction at the optimum, and the tests
check it only near the optimum.

## License

MIT
