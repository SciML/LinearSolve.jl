# BorderedLinearSolve.jl

Solvers for bordered linear systems

```
┌      ┐┌   ┐   ┌   ┐
│ J  b ││ v │ = │ f │
│ c' d ││ σ │   │ g │
└      ┘└   ┘   └   ┘
```

where `J` is `n x n` and the border `b`, `c` is `n x m` with `m` small. Systems of this
shape come up in continuation and bifurcation work, where `J` is the Jacobian of a large
system and the border carries a few extra constraints such as an arclength condition.

Built on [LinearSolve.jl](https://github.com/SciML/LinearSolve.jl), so the inner solves
pick up its full set of algorithms, preconditioners, and GPU support.

## Why not just assemble the matrix

You can form the whole `(n+m) x (n+m)` matrix and solve it, and for a small dense `J`
that is the fastest thing to do (`DirectBLS` does exactly this). It stops being a good
idea when `J` is large:

  - a sparse `J` gains a dense row and column, which the factorization has to carry
  - if `J` is matrix-free there is no matrix to assemble in the first place
  - the assembled matrix changes every outer step, so nothing can be reused

Block elimination instead solves against `J` itself, twice for a width-1 border. Both
solves use the *same* `J`, so one `LinearSolve` cache serves them, and it survives
between continuation steps where `J` changes only slightly. That is what makes
factorization reuse and Krylov subspace reuse available.

## Use

```julia
using BorderedLinearSolve, LinearSolve

prob = BorderedLinearProblem(J, b, c, d, f, g)
sol = solve(prob, BorderingBLS())
sol.v, sol.σ
```

`f` is the state and may be any array. For a 2D problem keep it as a matrix; `b` and
`c` then have the same shape, and the solution comes back in that shape too. Nothing
here asks you to `vec` and `reshape` around the solver.

`J` is only applied and solved against, so it can be a matrix or a matrix-free
operator, and no assumption is made about its shape.

A border wider than 1 is given explicitly as a tuple of states, with `d` an `m x m`
matrix and `g` of length `m`:

```julia
prob = BorderedLinearProblem(J, (b1, b2), (c1, c2), d, f, g)
```

The width is taken from that structure rather than inferred from an array's shape,
because a state can itself be a matrix and the two cannot otherwise be told apart.

Pick the inner algorithm the way you would for any `LinearSolve` problem:

```julia
solve(prob, BorderingBLS(KrylovJL_GMRES()))
solve(prob, BorderingBLS(LUFactorization()))
```

When the same `J` is solved against repeatedly, build the cache once and keep it. This
is the case the package is for:

```julia
cache = init(prob, BorderingBLS(LUFactorization()))
sol = solve!(cache)
# ... update J in place, then solve! again, reusing the factorization
```

## Algorithms

| | |
|---|---|
| `BorderingBLS(; inner, k, tol)` | block elimination, `k` correction sweeps |
| `DirectBLS(; inner)` | assemble `[J b; c' d]` and solve it |

`BorderingBLS` loses accuracy as `J` approaches singular, which is exactly where
continuation spends its time. The `k` correction sweeps recover it, at two extra inner
solves each. On a matrix with a `1e-11` singular value:

```
k = 0   relative error 6.8e-5
k = 3   relative error 1.9e-15
```

The default is `k = 0`; raise it if you are working near a fold.

## GPU

Runs under `CUDA.allowscalar(false)` with a `CuArray` or `CuSparseMatrixCSR` `J`. On a
T4, against a CPU reference, for `n = 200`:

| inner solver | relative error |
|---|---|
| default | 9.3e-16 |
| `LUFactorization` | 7.2e-16 |
| `KrylovJL_GMRES` | 7.0e-10 |
| `KrylovJL_GMRES`, sparse `J` | 1.6e-8 |

For a border wider than 1 the small `m x m` system is solved on the host, since `m` is
small by assumption and a device solve at that size would pivot scalar-wise.

Right-hand sides are copied before being handed to the inner cache. Some backends solve
in place, and on a GPU that was observed to overwrite the caller's `b`, which both
mutates the problem you passed in and corrupts the residual the correction sweeps
depend on.

## Reference

Govaerts, W. "Stable Solvers and Block Elimination for Bordered Systems." SIAM Journal
on Matrix Analysis and Applications 12, no. 3 (1991): 469-83.

## Status

Early. Written in response to
[SciML/LinearSolve.jl#560](https://github.com/SciML/LinearSolve.jl/issues/560). Feedback
on the interface is welcome.
