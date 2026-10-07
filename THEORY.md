# Theory: where a wormhole ends, and what it can never do

The experiments in the README say that a transform `T` changes the route of gradient descent,
not the destination, except when many answers fit. This note makes that exact for the case where
many answers fit, and derives a ceiling on what any wormhole can recover. The first result is
the known equivalence between reparameterized gradient descent and mirror descent (Gunasekar et
al. 2018; Amid and Warmuth 2020; Li et al., NeurIPS 2022). The ceiling is a short consequence of
it. E11 and E12 in `python -m charon.frontier` check both numerically.

## Setting

`n` equations, `d > n` unknowns, `X w = y` has many solutions. The loss is
`L(w) = |X w − y|² / (2n)`. A wormhole is a metric `m > 0`, the same for every coordinate and
even (`m(w) = m(−w)`), and training is the flow

```
dw_i/dt = −m(w_i) · ∂L/∂w_i
```

Every elementwise transform `w = T(θ)` trained by gradient descent on `θ` is this flow, with
`m(w) = T'(T⁻¹(w))²` (charon's one-step identity). styx learns `m` directly.

Define the potential `φ(w) = ∫₀ʷ ∫₀ᵘ ds / m(s) du`. Then `φ'' = 1/m > 0`, so `φ` is strictly
convex and even, with `φ(0) = φ'(0) = 0` whenever `1/m` is integrable at 0.

## Theorem 1: the destination does not depend on the route

*If the flow starts at `w(0) = 0` and converges to an exact fit `w∞`, then*

```
w∞ = argmin  Σ_i φ(w_i)   subject to   X w = y.
```

*Proof.* Along the flow, `d/dt φ'(w_i) = φ''(w_i) · dw_i/dt = −∂L/∂w_i`. Stacked over
coordinates, `d/dt φ'(w) = −Xᵀ(Xw − y)/n`, which lies in the row space of `X`. Integrating,
`φ'(w∞) − φ'(0) = Xᵀν` for some `ν`. With `φ'(0) = 0`, this is exactly the optimality
condition of the convex problem above, `∇Σφ(w) = Xᵀν` with `Xw = y`. `φ` is strictly convex,
so the minimizer is unique. ∎

Three consequences:

- **Only the shape matters, not the speed.** Multiplying `m` by any constant multiplies `φ` by
  its inverse and leaves the minimizer alone. A faster wormhole takes a different route to the
  same place.
- **The destination can be computed without training.** The optimality condition says
  `w = ψ(Xᵀν)`, where `ψ` is the inverse of `φ'`. That leaves `n` unknowns, and `destination()`
  in [charon/destination.py](charon/destination.py) finds them by Newton's method on the
  convex dual. E11: gradient descent lands on the predicted point. The gap is proportional to
  the step size (for `u² − v²`: 2.4e-2 at lr 0.25, 2.6e-3 at 0.025, 2.6e-4 at 0.0025), the
  discretization error of a flow.
- **Dead points.** If `m(0) = 0`, `w = 0` is a resting point of the flow. Gradient descent
  started at exactly zero never moves, even when `φ` is finite. This is charon's `w²` "dead point
  at w = 0" in general form.

## Proposition: a wormhole with `m(0) > 0` never recovers a sparse vector exactly

If `m(0) > 0` then `φ'(0) = 0`, and `x*` with support `S` is the destination only if
`φ'(x*) ∈ row space of X`. That vector is nonzero on `S` and zero elsewhere. For `X` with
random Gaussian entries, an `n`-dimensional subspace contains no nonzero vector that vanishes on
`d − |S| > n` fixed coordinates (with probability one). So the destination is never exactly
sparse. It only approaches sparsity as the wormhole's weight near zero shrinks. charon's E6
saw exactly this. The `u² − v²` answer reached the sparse truth only as its starting size went
to zero.

## Theorem 2: the ceiling

Take the limit of a family of wormholes in which `φ` keeps a kink at zero. That is, `φ` is even
and convex with `φ(0) = 0`, right-derivative `c = φ'(0⁺) > 0` at zero, and `φ'` nondecreasing on
`(0, ∞)`. Every limit of wormholes has this shape, because `φ'' = 1/m > 0` makes `φ'`
nondecreasing. L1 is the case where `φ'` is constant.

*Fix a support `S` and a sign pattern `s`. If `Σφ` recovers one vector `x*` with support `S`,
signs `s` and equal magnitudes (that is, `x*` minimizes `Σφ(w)` subject to `Xw = Xx*`), then L1
minimization recovers every vector with support `S` and signs `s`.*

*Proof.* Let `N` be the null space of `X`, and `t` the common magnitude. `x*` minimizes the
convex `Σφ` on `x* + N` exactly when every directional derivative is nonnegative:

```
a · Σ_{i∈S} s_i h_i  +  Σ_{i∉S} |h_i|  ≥  0     for all h in N,     a = φ'(t) / c ≥ 1.
```

(Divide by `c`; `a ≥ 1` because `φ'` is nondecreasing.) Take any `h ∈ N`. If
`Σ_S s_i h_i ≥ 0`, the L1 condition `Σ_S s_i h_i + Σ_{S^c} |h_i| ≥ 0` holds trivially.
Otherwise `Σ_{S^c} |h_i| ≥ −a Σ_S s_i h_i ≥ −Σ_S s_i h_i`, so it holds again. The L1
condition depends only on `S` and `s`, so L1 recovers every vector with that support and
those signs. ∎

So a wormhole can't beat L1 on the problems L1 finds hardest: equal-size nonzeros. Uniformly
over sizes, it can at best tie. When the nonzeros differ in size, `a` varies across the support,
and the theorem says nothing about single instances. E12 counts them. Among 800 problems, both
equal-size and uniformly sized, with 6 to 12 nonzeros, it records how often a wormhole recovers
a vector that L1 misses.

## The ways out

The theorem leaves every one of its assumptions open, and each is a way around the ceiling:

1. **A metric that changes with time.** Then there is no fixed `φ`, and the destination depends
   on the route. Annealing `τ` in styx's log wormhole, one such attempt, made things worse.
2. **A metric that couples coordinates**, a full matrix instead of an elementwise `m`.
3. **A start away from zero.** The destination becomes the Bregman projection of `w(0)`,
   which can carry prior knowledge.
4. **Large steps and noise.** These break the flow approximation. Large-step "drift" in diagonal
   networks is studied by Sanyal, Jacobs and Burkholz (2026, arXiv 2610.06120).
5. **Nonlinear models**, where no linear program exists and the convex picture is gone. styx's
   network results live here.

On linear problems, an elementwise wormhole that stays fixed during training is a separable
convex penalty in disguise. Learning one, as styx does, learns that penalty. Theorem 2 says that
such a penalty can at best tie L1, uniformly over sizes.

## The quantum case: non-commuting wormholes

Replace the vector by a density matrix `ρ`: Hermitian, positive semidefinite, trace 1. The
data are Pauli expectation values `tr(P_k ρ)`, fewer of them than the `d² − 1` unknowns. Two
matrix wormholes:

- **Born**, `ρ = AA†` with `A` a full `d × d` matrix, the matrix version of `u²`. Gradient
  descent on `A` does not commute with `ρ`, so it is not a mirror flow in general (Li, Wang,
  Lee and Arora, 2022, show that some flows of `UUᵀ` are no mirror flow at all). From a small
  start it builds `ρ` greedily one rank at a time (Li, Luo and Lyu, ICLR 2021).
- **Gibbs**, `ρ = exp(H)/tr exp(H)`. Gradient steps on `H` are mirror descent with the von
  Neumann entropy, matrix exponentiated gradient (Tsuda, Rätsch and Warmuth, 2005). From
  `H = 0` this is the maximum-entropy principle.

## Theorem 3: convex, basis-free penalties prefer the most mixed state

*Let `f` be convex on density matrices and unitarily invariant (`f(UρU†) = f(ρ)` for every
unitary `U`). Then `f(I/d) ≤ f(ρ)` for every state `ρ`.*

*Proof.* Averaging `UρU†` over the Haar measure gives `I/d` (the twirl). By Jensen,
`f(I/d) = f(∫UρU†dU) ≤ ∫f(UρU†)dU = f(ρ)`. ∎

Among the states that fit the data, a convex penalty that ignores the basis can therefore only
pull toward the mixed end. Von Neumann entropy, the nuclear norm (always 1 here), the
Frobenius norm and every Schatten norm are all of this kind. None of them can prefer a
purer state. The commuting theory makes every fixed elementwise wormhole a convex penalty
(Theorem 1), so a preference for purity has to come from somewhere else. The Born wormhole
supplies it through non-commuting, non-convex dynamics. E13 and E14 measure what that
preference buys and what it costs.
