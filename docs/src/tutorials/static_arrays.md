# Solving with StaticArrays

`LinearProblem`s whose `A` and `b` are
[StaticArrays](https://github.com/JuliaArrays/StaticArrays.jl) (`SMatrix`,
`SVector`) take a dedicated non-caching fast path: the solve happens directly
on the immutable arrays with **zero heap allocations**, and the returned
solution's `u` is a static array of the matching shape.

```@example static
using LinearSolve, StaticArrays

A = SA[2.0 1.0; 1.0 3.0]
b = SA[1.0, 2.0]
prob = LinearProblem(A, b)

sol = solve(prob)
sol.u
```

Because static problems are immutable, there is no `init!`/`solve!` caching
interface for them — each `solve` is a self-contained direct solve. (There is
also nothing to cache: no factorization object is retained.)

## Algorithm choices and singular-input behavior

Static square problems support a tier of algorithms trading safety against
overhead. All of them are allocation-free; timings below are for a 2×2
`Float64` system on a prebuilt `LinearProblem` (bare `A \ b` is 4.4 ns on the
same machine) and scale with the usual method costs at larger sizes.

| algorithm | ~2×2 cost | on singular `A` |
|---|---|---|
| `DirectLdiv!()` | 5.6 ns | unchecked: whatever `\` gives — non-finite values for `N ≤ 3`, a thrown `SingularException` for larger `N` |
| `GESVFactorization()` | 8–9 ns | `ReturnCode.Failure` with zeroed `u`; no rescue (one-shot factorize-and-solve, the static analog of LAPACK's `gesv` driver) |
| `LUFactorization()` | 22 ns | `ReturnCode.Failure` with zeroed `u`, matching the dense `LUFactorization` behavior |
| default (`solve(prob)`) | 9.2 ns | rescued: an SVD least-squares min-norm pseudo-solution with `ReturnCode.Success`, mirroring the dense default's singular-LU → pivoted-QR safety fallback |

Guidance:

  - Use the **default** unless you have a reason not to: it is nearly as fast
    as the unchecked path (the finiteness check fuses with the solve) and a
    singular matrix degrades gracefully instead of silently producing `Inf`/
    `NaN`.
  - Use **`DirectLdiv!`** when the matrix is known nonsingular and every
    nanosecond counts: it is a bare `A \ b` behind one algorithm-type branch.
  - Use **`GESVFactorization`** when you want failure *reported* (a checkable
    return code) but not *repaired* — e.g. inside an iteration that has its own
    recovery logic. It uses the direct small-size kernels, so it is faster than
    `LUFactorization` at small sizes.
  - `LUFactorization` on static arrays exists for algorithm-genericity (code
    that passes an algorithm through to both dense and static problems); the
    static `lu` costs about 3× the direct inverse formulas at `N ≤ 3`.

`QRFactorization`, `CholeskyFactorization`, `NormalCholeskyFactorization`, and
`SVDFactorization` also have direct static dispatches with their usual
applicability requirements.

Nonsingular results from the checked algorithms match `A \ b` up to the last
bit or ulp-level differences from inlining-dependent FMA contraction in
StaticArrays' small-size kernels.

## Non-square static problems

Non-square static problems route to an SVD least-squares solve by default
(static QR least-squares division is not available in StaticArrays), returning
the minimum-norm solution:

```@example static
Ans = SA[1.0 2.0; 3.0 4.0; 5.0 6.0]
bns = SA[1.0, 2.0, 3.0]
solve(LinearProblem(Ans, bns)).u
```

`GESVFactorization` is square-only and throws an `ArgumentError` for non-square
static input, matching the LAPACK driver it mirrors.
