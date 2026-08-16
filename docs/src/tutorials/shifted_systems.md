# Repeated Solves of a Shifted System

Some problems ask you to solve the *same* matrix, shifted many different ways:

```math
(\sigma I + \tau J)\, x = b
```

for one `J` and a long sequence of scalars. The archetype is the iteration matrix of an
implicit ODE solver, `W = I - γJ` with `γ = c·dt`: adaptive step-size
control moves `γ` on nearly every step, while the Jacobian `J` is held fixed for tens
of steps because recomputing it is expensive. Sweeping a resolvent `(sI - A)⁻¹` over `s`,
or running shift-and-invert with a moving shift, have the same shape.

Factorizing from scratch each time pays `O(n³)` to absorb a scalar. This page is about not
doing that.

## The naive loop, and the cost of it

Take a stiff Jacobian and ten shifts:

```@example shifted
import LinearSolve as LS
import LinearAlgebra as LA
using Random

Random.seed!(42)
n = 400
J = randn(n, n) - 20LA.I      # something with a stiff, decaying spectrum
b = randn(n)
γs = range(0.01, 0.05, length = 10)

function refactorize_each_time(J, b, γs)
    local x
    for γ in γs
        x = (LA.I - γ * J) \ b
    end
    return x
end

refactorize_each_time(J, b, γs)                        # warm up the compiler
naive_ms = 1000 * (@elapsed refactorize_each_time(J, b, γs))
```

Every iteration builds a fresh `n × n` matrix and runs a fresh LU. Nothing about `J` is
reused, even though `J` never changed.

## Handing the solver the split form

The fix is to give the solver `J` and the shift *separately*, so it can see that only the
scalar moved. LinearSolve spells that with `SciMLOperators.WOperator`, which holds a mass
matrix, a `gamma` and a Jacobian without assembling them, and
[`LHLFactorization`](@ref), the algorithm that exploits the split:

```@example shifted
import SciMLOperators as SO

W = SO.WOperator{true}(LA.I, 0.01, J, similar(b))    # represents J - I/0.01
cache = LS.init(LS.LinearProblem(W, b), LS.LHLFactorization())
sol = LS.solve!(cache)
sol.u ≈ (J - LA.I / 0.01) \ b
```

A `WOperator` stands for `J - M/γ`, the "W-transform" an implicit solver actually
forms, rather than `I - γJ`. The two differ by a factor `-γ`, so they have the
same solution up to scaling the right-hand side; pick whichever your problem states
naturally and set `gamma` accordingly.

Now move the shift with [`update_gamma!`](@ref) instead of rebuilding anything:

```@example shifted
function reuse_one_reduction(cache, γs)
    local u
    for γ in γs
        LS.update_gamma!(cache, γ)
        u = LS.solve!(cache).u
    end
    return u
end

reuse_one_reduction(cache, γs)                         # warm up the compiler
split_ms = 1000 * (@elapsed reuse_one_reduction(cache, γs))
(naive_ms, split_ms, naive_ms / split_ms)
```

The first solve paid for a reduction of `J`; every later `γ` costs `O(n²)`. The ratio above
is whatever the machine that built these docs measured over ten shifts; it grows with both
`n` and the number of shifts.

## Why it works

`LHLFactorization` reduces `J` once to upper Hessenberg form by a similarity transform:

```math
J = Z H Z^{-1}
```

Because it is a *similarity*, the shift passes straight through it:

```math
\sigma I + \tau J = Z\,(\sigma I + \tau H)\,Z^{-1}
```

`Z` carries no `γ`, so it is computed once and never touched again. What is left,
`σI + τH`, is Hessenberg — one subdiagonal — and its LU costs `O(n²)` rather
than `O(n³)`. Solving is then `Z^{-1}`, a Hessenberg solve, and `Z`.

The reduction is done by Gaussian elimination-style similarity transformations with partial
pivoting, giving `Z = D·P·L` with `L` unit lower triangular, `P` a permutation
and `D` a balancing diagonal — hence *LHL*. It lives in
[LHLFactorization.jl](https://github.com/SciML/LHLFactorization.jl) and can be used
directly, without LinearSolve, if you want the kernels rather than the solver interface.

### Three phases at three different frequencies

|  | how often | cost |
|---|---|---|
| reduce `J` | once per Jacobian | `5/3 n³` |
| re-shift | once per `γ` | `n²` |
| solve | once per right-hand side | `≈` an LU's triangular solves |

Measured against LAPACK on one thread, a re-shift is 14× cheaper than a refactorization at
`n = 25` and 108× at `n = 800`. That is the whole trade: an up-front reduction, roughly
2.5× an LU, bought back by every shift after the first.

## Telling the solver when `J` changes

Only `γ` is visible to the cache — it is a field, so a change is self-evident. Writing
new numbers *into* `J` is not visible: same object, same size, same everything. Say so
explicitly:

```@example shifted
J .= randn(n, n) - 20LA.I
SO.mark_jacobian_updated!(W)          # next solve redoes the O(n³) reduction
LS.solve!(cache).u ≈ (J - LA.I / 0.01) \ b
```

Forget this and the cache will happily reuse a reduction of the old Jacobian. Swapping in a
*different* `WOperator` needs no announcement — the cache notices the matrix is not the one
it reduced.

!!! warning "One consumer per operator"
    `mark_jacobian_updated!` sets a flag that the consumer clears once it has refactorized.
    Two caches sharing a single `WOperator` will race: whichever refactorizes first clears
    the flag and the other never learns. Give each consumer its own operator.

## Accuracy, and the `refine` keyword

`Z` is not orthogonal, so unlike an LU the backward error carries a factor `κ(Z)`.
Partial pivoting keeps the elimination well behaved, but near-nilpotent or badly scaled
Jacobians can still lose eight digits or more.

One step of iterative refinement removes that, and it is the default (`refine = 1`). It
costs roughly 1.4–1.6 solves and brings the forward error within 10× of LU's across the
stability study the algorithm was validated on.

```@example shifted
Jhard = LA.triu(randn(120, 120), 1)   # strictly upper triangular: every pivot near zero
Jhard[end, 1] = 1e-8
bh = randn(120)
Wh = SO.WOperator{true}(LA.I, 0.5, Jhard, similar(bh))
exact = (Jhard - LA.I / 0.5) \ bh

raw = LS.solve(LS.LinearProblem(Wh, bh), LS.LHLFactorization(refine = 0)).u
refined = LS.solve(LS.LinearProblem(Wh, bh), LS.LHLFactorization(refine = 1)).u
(LA.norm(raw - exact) / LA.norm(exact), LA.norm(refined - exact) / LA.norm(exact))
```

**Inside a Newton loop, consider `refine = 0`.** An implicit ODE solver runs on the order of
fourteen solves per `γ`, so refinement is charged against exactly the quantity the
algorithm is saving. Newton is itself a correction loop whose convergence test governs the
answer, so an inexact linear solve costs at most an extra iteration. The default stays at 1
because a bare linear solve has no outer loop to lean on.

## A complex shift on a real Jacobian

RadauIIA and friends need `(I - γ_c·h·J)` for a *complex* `γ_c` alongside a real
one, with the same real `J`. The reduction does not have to go complex for that — only the
shifted Hessenberg does, which keeps the expensive part real:

```@example shifted
γc = 0.03 + 0.02im
bc = randn(ComplexF64, n)
Wc = SO.WOperator{true}(LA.I, γc, J, zeros(ComplexF64, n))
uc = LS.solve(LS.LinearProblem(Wc, bc), LS.LHLFactorization()).u
uc ≈ (J - LA.I / γc) \ bc
```

## When it is chosen automatically

Wrapping a matrix in a `WOperator` is itself the statement that the shift will move while
`J` stays put, so `defaultalg` treats it as one: for a `WOperator` with a dense Jacobian, a
scalar multiple of `I` as the mass matrix, and `n ≥ LinearSolve.LHL_DEFAULT_MIN_SIZE`, no
algorithm argument is needed.

```@example shifted
LS.defaultalg(W, b, LS.OperatorAssumptions(true))
```

Below that size, or for a matrix-free Jacobian, the operator falls through to whatever it
would otherwise have got — a small problem is faster with a plain LU, and a matrix-free one
still wants a Krylov method.

## Limits

  - **Dense Jacobians only.** The Hessenberg reduction fills in, so sparsity buys nothing;
    a sparse Jacobian is rejected rather than silently densified.
  - **The mass matrix must be a multiple of `I`.** A general one needs a
    Hessenberg–triangular reduction of the pencil, which is not implemented.
  - **One workspace, one solve at a time.** The cache writes scratch buffers, so a single
    cache cannot serve concurrent solves.
  - **Not a general `Ax = b` solver.** Against a single system an LU wins on every axis. The
    reduction only pays because it is amortized over many shifts.
  - **Other factorizations cannot consume an in-place `WOperator`** over a plain-matrix
    Jacobian: its concrete form is maintained by the operator's owner rather than rebuilt on
    conversion, so it is stale after any change of `gamma`. LinearSolve raises rather than
    factorizing the wrong matrix.

## API

```@docs
LHLFactorization
update_gamma!
```
