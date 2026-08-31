# [Low-Rank Updates of a Factorized Matrix](@id lowrankupdates)

Some problems hand you a matrix that is cheap to factorize, then perturb it in a way that is
not cheap at all:

```math
(A + U C V^{*})\, x = b
```

`A` is sparse, banded, or otherwise structured, so its factorization costs far less than
`O(n³)`. The update `U C V^{*}` has rank `k ≪ n`, but it is *dense*. Adding the two together
throws away everything that made `A` cheap: the sum is a dense `n × n` matrix, and you pay a
dense factorization for a correction of rank `k`.

This page is about not assembling that sum.

## The cost of assembling

Take a tridiagonal `A` and a rank-1 update:

```@example lowrank
import LinearSolve as LS
import LinearAlgebra as LA
import SparseArrays as SA
using Random

Random.seed!(136)
n = 200
A = SA.spdiagm(-1 => -ones(n - 1), 0 => 4ones(n), 1 => -ones(n - 1))
u = rand(n)
v = rand(n)
b = rand(n)

assembled = Matrix(A) + u * v'
(SA.nnz(A), count(!iszero, assembled))
```

`A` holds 598 nonzeros. The sum holds all 40000 of them: one rank-1 term has filled in every
structural zero. The factorization cost follows:

```@example lowrank
LA.lu(A)                                   # warm up the compiler
LA.lu(assembled)
sparse_ms = 1000 * @elapsed LA.lu(A)
dense_ms = 1000 * @elapsed LA.lu(assembled)
(sparse_ms, dense_ms, dense_ms / sparse_ms)
```

At `n = 200` the gap is already most of an order of magnitude, and it widens with `n`: the
sparse factorization of a banded matrix is `O(n)` here while the dense one is `O(n³)`. The
update contributed `2n` numbers and cost you the entire structure.

## Wrapping the update instead

[`LowRankUpdatedMatrix`](@ref) carries `A + U C V^{*}` in that form rather than assembling
it. It is a matrix type, not an algorithm, so it goes where the matrix goes: pass it as the
matrix of a `LinearProblem` and solve normally.

```@example lowrank
M = LS.LowRankUpdatedMatrix(A, u, v)
sol = LS.solve(LS.LinearProblem(M, b))
(sol.retcode, LA.norm(sol.u - assembled \ b) / LA.norm(sol.u))
```

Same answer, to rounding. What changed is the arithmetic underneath. The solve factorizes
`A` alone, keeping whatever structure `A` has, and reaches the answer through the Woodbury
identity:

```math
(A + U C V^{*})^{-1} = A^{-1} - A^{-1} U \left(C^{-1} + V^{*} A^{-1} U\right)^{-1} V^{*} A^{-1}
```

Every appearance of `A` on the right is an `A⁻¹` applied to a vector or to the `n × k` block
`U`, which the factorization of `A` already gives you. The only new factorization is of the
`k × k` *capacitance matrix* `C⁻¹ + V* A⁻¹ U`. A rank-1 update therefore costs one extra
solve against `A` and the inversion of a `1 × 1` matrix, not a fresh `n × n` factorization.

## Rank-k updates and the middle factor

`U` and `V` are `n × k` and `C` is `k × k`. A vector `U` or `V` is read as a rank-1 update,
which is the case above. `C` defaults to `I`, giving the Sherman-Morrison form `A + U V^{*}`.

Supplying `C` is how you write an update that is naturally a product of three factors, for
example a correction `U C V^{*}` where `C` carries the scaling and `U`, `V` hold the
directions:

```@example lowrank
m = 40
k = 3
Ad = rand(m, m) + m * LA.I
U = rand(m, k)
V = rand(m, k)
C = rand(k, k) + k * LA.I
bd = rand(m)

Mk = LS.LowRankUpdatedMatrix(Ad, U, V; C = C)
dense_k = Ad + U * C * V'
LS.solve(LS.LinearProblem(Mk, bd)).u ≈ dense_k \ bd
```

The three factors are checked against each other at construction, so a shape error surfaces
where you wrote it rather than inside a solve:

```@example lowrank
try
    LS.LowRankUpdatedMatrix(Ad, rand(m, 2), rand(m, 3))
catch err
    err isa DimensionMismatch
end
```

!!! note "Rank is what pays"
    The capacitance matrix is `k × k`, so the saving is a function of `k` against `n`, not of
    how the update was written. At `k` comparable to `n` the Woodbury route costs more than
    refactorizing.

## It is a matrix

`LowRankUpdatedMatrix <: AbstractMatrix`, and it implements enough of the interface that
generic code does not have to know it is a special type: `size`, `size(M, i)`, `eltype`,
`getindex`, and `LinearAlgebra.mul!`.

```@example lowrank
x = rand(m)
y = similar(x)
LA.mul!(y, Mk, x)

(size(Mk), size(Mk, 1), eltype(Mk),
    Mk[3, 7] ≈ dense_k[3, 7],
    y ≈ dense_k * x,
    Mk * x ≈ dense_k * x)
```

`mul!` is the important one. It applies `A` and then adds `U (C (V* x))`, right to left,
never forming the outer product. That makes the type usable as the operator of a Krylov
method, or anywhere else that only multiplies.

`getindex` is there for convenience and for printing. It reconstructs one entry at a time
from a row of `U` and a row of `V`, so indexing in a hot loop is the wrong way to use this
type. `Matrix(M)` works and gives you the assembled sum, which is exactly what the type
exists to avoid.

## Reusing the factorization across right-hand sides

The factorization of `A`, the block `A⁻¹U`, and the factorized capacitance matrix are all
independent of `b`. They are built on the first solve and kept, so the caching interface
turns every later right-hand side into a pair of triangular solves plus a `k × k` solve:

```@example lowrank
cache = LS.init(LS.LinearProblem(M, b))
LS.solve!(cache)

worst = maximum(1:4) do _
    bnew = rand(n)
    cache.b = bnew
    LA.norm(LS.solve!(cache).u - assembled \ bnew) / LA.norm(bnew)
end
```

This is where the saving compounds. One factorization of `A` and one `k × k` factorization
serve the whole sequence. Assigning a new matrix to the cache marks it stale and rebuilds
both.

## Choosing the inner factorization

The algorithm you pass factorizes `A`, not the sum, so pick it to suit `A`. With no
algorithm given, `defaultalg` picks LU:

```@example lowrank
LS.defaultalg(M, b, LS.OperatorAssumptions(true))
```

That is the right default for the motivating case: `lu` dispatches on the type of `A`, so a
sparse `A` gets the sparse kernel and a dense `A` gets LAPACK. To choose explicitly, pass
the algorithm as usual:

```@example lowrank
[LS.solve(LS.LinearProblem(Mk, bd), alg).u ≈ dense_k \ bd
 for alg in (LS.LUFactorization(), LS.QRFactorization(), LS.SVDFactorization())]
```

The set of algorithms that take this route is enumerated in `LinearSolve._LOWRANK_ALGS`, not
dispatched generically. Exactly these are supported:

| Algorithm | Use it when |
|---|---|
| [`LUFactorization`](@ref) | the default; dense or sparse `A` |
| [`QRFactorization`](@ref) | `A` is ill-conditioned and you want the extra stability |
| [`CholeskyFactorization`](@ref) | `A` is symmetric positive definite |
| [`BunchKaufmanFactorization`](@ref) | `A` is symmetric indefinite, wrapped in `Symmetric` |
| [`SVDFactorization`](@ref) | `A` is near-singular and you want the most careful option |

Two things are worth reading off that table. The first is that it constrains **`A`**, not
the sum: `A` can be symmetric while `A + U C V^{*}` is not, since nothing requires `V` to
equal `U`. Splitting the two puts the structure the factorization needs in the part being
factorized. The second is that `LUFactorization` covers the sparse case on its own, because
`lu` dispatches on the type of `A` and picks the sparse kernel. That is why
`UMFPACKFactorization` and `KLUFactorization` are absent: they reach their factorizations
through their own `solve!` rather than through `do_factorization`, which is the hook the
correction uses.

A matrix-free method needs none of this. `mul!` is defined on the wrapper, so a Krylov
solver consumes it directly, never assembles it, and simply does not use the identity:

```@example lowrank
LS.solve(LS.LinearProblem(Mk, bd), LS.KrylovJL_GMRES()).u ≈ dense_k \ bd
```

!!! warning "A factorization outside the list will raise"
    Passing a factorization that is neither in the table nor matrix-free does not silently
    assemble the sum: it reaches a kernel that cannot consume the wrapper and raises. That
    is deliberate. Use one of the listed factorizations, a Krylov method, or assemble the
    matrix yourself with `Matrix(M)` if you genuinely want the dense path.

## When the update is singular

The Woodbury identity needs both `A` and the capacitance matrix to be nonsingular. A rank-1
update that cancels a diagonal entry does not disturb `A` at all, so the failure shows up in
the capacitance matrix:

```@example lowrank
Asing = Matrix{Float64}(LA.I, 4, 4)
usg = [1.0, 0.0, 0.0, 0.0]
vsg = [-1.0, 0.0, 0.0, 0.0]

singular = LS.solve(LS.LinearProblem(LS.LowRankUpdatedMatrix(Asing, usg, vsg), rand(4)))
singular.retcode
```

Here `A + u v^{*}` is `diag(0, 1, 1, 1)`, genuinely singular. The solve logs the failure and
returns a non-success retcode rather than dividing by a near-zero pivot and handing back a
plausible-looking vector. Check `sol.retcode` for the same reason you would with any other
factorization: a singular system is a fact about your problem, and the solver's job is to
say so.

The same reporting covers the case where `A` is fine but the update is not. `A` and the
capacitance matrix are factorized separately, so a failure of either one produces the same
non-success retcode.

## Limits

  - **The update must be genuinely low rank.** The capacitance matrix is `k × k` and gets a
    dense factorization. Once `k` approaches `n` you are factorizing something nearly as big
    as the original, on top of factorizing `A`, and refactorizing the sum is cheaper.
  - **The capacitance matrix must be nonsingular.** `C⁻¹ + V* A⁻¹ U` singular means either
    the updated matrix is singular or the update is written in a degenerate way. Either way
    the identity does not apply, and the solve fails rather than guessing.
  - **`A` itself must be nonsingular**, even when `A + U C V^{*}` is well conditioned. The
    identity works through `A⁻¹`, so an update that repairs a singular `A` is out of reach
    here, though a direct factorization of the sum would handle it.
  - **Woodbury is less stable than refactorizing.** The identity is a difference of two
    terms, so an ill-conditioned update makes it a subtraction of nearly equal quantities.
    The accuracy is governed by the conditioning of `A` and of the capacitance matrix, not by
    that of `A + U C V^{*}`, and those can be far apart. When accuracy matters more than
    speed and the update is nasty, assemble and factorize.
  - **Only the enumerated factorizations take the fast path.** The table above is the whole
    list. Iterative and matrix-free algorithms do not use the identity, though `mul!` means
    they can still consume the type as an operator.
  - **The update is fixed once constructed.** Changing `U`, `V`, or `C` means building a new
    `LowRankUpdatedMatrix` and assigning it to the cache. The base factorization is reused
    across right-hand sides, not across different updates.

## Reference

[`LowRankUpdatedMatrix`](@ref), the supported factorizations, and the cost breakdown are
documented on the [Low-Rank Updated System Solvers](@ref lowranksolvers) page. The
factorizations themselves are listed with everything else on the
[Linear System Solvers](@ref linearsystemsolvers) page. For the caching interface this
tutorial leans on, see [Linear Solve with Caching Interface](@ref).
