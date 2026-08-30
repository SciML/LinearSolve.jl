# Linear Solver Algorithm Interface

Every algorithm that LinearSolve.jl can be handed as `solve(prob, alg)` is a
subtype of `LinearSolve.SciMLLinearSolveAlgorithm`. This page is the contract
that such a type has to satisfy. It applies equally to algorithms defined inside
LinearSolve.jl, in one of its package extensions, and in downstream packages.

The interface is checked in LinearSolve.jl's test suite for every algorithm the
package knows about, using [`LinearSolve.algorithm_interface_issues`](@ref);
downstream packages can run the same check against their own algorithms.

## Choosing a supertype

```
SciMLLinearSolveAlgorithm
├── AbstractFactorization
│   ├── AbstractDenseFactorization
│   └── AbstractSparseFactorization
├── AbstractKrylovSubspaceMethod
└── AbstractSolveFunction
```

Subtype the most specific abstract type that fits. Doing so supplies several of
the methods below; subtyping `SciMLLinearSolveAlgorithm` directly supplies none
of them, so such an algorithm must define `needs_concrete_A` itself. Every trait
default that comes from the categorized types is listed in the tables below.

## Required methods

| Method | Meaning |
|:--- |:--- |
| `SciMLBase.solve!(cache::LinearCache, alg::MyAlg; kwargs...)` | Solve the system held in `cache` and return `SciMLBase.build_linear_solution(alg, u, resid, cache)`. |
| `LinearSolve.needs_concrete_A(alg::MyAlg)::Bool` | Whether the algorithm needs the entries of `A`, or only matrix-vector products with it. |

`solve!` is the algorithm proper. It reads `cache.A`, `cache.b`, `cache.u`,
`cache.Pl`/`cache.Pr`, `cache.abstol`/`cache.reltol`/`cache.maxiters`, and
writes the solution into `cache.u`. `cache.isfresh` is `true` when `A` has
changed since the previous solve; algorithms that build a factorization or a
solver object should do so only when it is set, store the result in
`cache.cacheval`, and then set `cache.isfresh = false`.

`needs_concrete_A` is a pure function of the algorithm type. Downstream solvers
— OrdinaryDiffEq.jl and NonlinearSolve.jl in particular — call it on a
user-supplied `linsolve` to decide whether to assemble a concrete Jacobian.

!!! warning "Define traits next to the struct, never in an extension"
    
    `needs_concrete_A`, `needs_square_A`, `default_alias_A` and
    `default_alias_b` must be defined in the same package (and ideally the same
    file) as the algorithm struct, not in the package extension that implements
    the algorithm. They are queried before the backend package is necessarily
    loaded, so a trait that lives in an extension is either missing (a
    `MethodError` deep inside an unrelated solver) or silently wrong (the
    inherited default is used instead of the intended value). Only `solve!`,
    `init_cacheval`, `update_tolerances_internal!`, and other methods that
    genuinely need the backend belong in the extension.
    
    `algorithm_interface_issues` reports a trait defined in an extension as a
    violation, so this is enforced rather than merely recommended.

## Optional methods

Each of these has a default, so an algorithm only defines the ones whose default
does not fit.

| Method | Default | Purpose |
|:--- |:--- |:--- |
| `LinearSolve.init_cacheval(alg, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assumptions)` | `nothing` | Build the algorithm's private cache, stored as `cache.cacheval`. Should return the same type that `solve!` later stores, so the cache stays type-stable. |
| `LinearSolve.default_alias_A(alg, A, b)::Bool` | `false`; `true` for `AbstractKrylovSubspaceMethod` and `AbstractSparseFactorization` | Whether `init` may alias the user's `A` instead of copying it. `true` is only correct for algorithms that never mutate `A`. |
| `LinearSolve.default_alias_b(alg, A, b)::Bool` | `false`; `true` for `AbstractKrylovSubspaceMethod` and `AbstractSparseFactorization` | The same for `b`. |
| `LinearSolve.needs_square_A(alg)::Bool` | `true` | Whether the algorithm requires a square `A`. Least-squares capable algorithms return `false`. |
| `LinearSolve.update_tolerances_internal!(cache, alg, abstol, reltol)` | throws: the algorithm has no tolerances to update | Hook for [`LinearSolve.update_tolerances!`](@ref). `LinearCache.abstol`/`reltol` have already been set when it runs. |

`update_tolerances_internal!` needs no definition when the algorithm has no
tolerances at all. Define it as `nothing` when the algorithm reads
`cache.abstol`/`cache.reltol` at solve time (Krylov methods do), and define it
to write into `cache.cacheval` when the algorithm snapshots the tolerances into
its own solver object at `init` time.

The four `Bool`-valued traits are called while `init` builds the cache, so they
must be inferable as `Bool` — write them as literal returns
(`needs_concrete_A(::MyAlg) = true`) rather than as runtime computations.

## A complete example

```julia
using LinearSolve, LinearAlgebra, SciMLBase

struct MyLUFactorization <: LinearSolve.AbstractDenseFactorization end

# Optional: give the cache its final type up front so repeated solves stay
# type-stable. `ArrayInterface.lu_instance(A)` is the cheap way to do this.
function LinearSolve.init_cacheval(
        alg::MyLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose, assump::LinearSolve.OperatorAssumptions
    )
    return lu(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::MyLUFactorization; kwargs...)
    if cache.isfresh
        cache.cacheval = lu!(convert(AbstractMatrix, cache.A))
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval, cache.b)
    return SciMLBase.build_linear_solution(alg, y, nothing, cache)
end
```

Subtyping `AbstractDenseFactorization` supplies `needs_concrete_A(alg) = true`
and the `false` aliasing defaults, which is what a destructive dense
factorization wants. An algorithm that subtypes `SciMLLinearSolveAlgorithm`
directly would additionally need

```julia
LinearSolve.needs_concrete_A(::MyLUFactorization) = true
```

## Matrix-free implementation

An iterative algorithm can promise that it only needs the generic `mul!`
operation. The operator does not need to expose a factorization or a dense
matrix representation; the algorithm still receives the standard
`LinearCache` and returns the standard `LinearSolution`.

```julia
struct DocIdentityOperator{T} <: AbstractMatrix{T}
    n::Int
end

Base.size(A::DocIdentityOperator) = (A.n, A.n)
Base.getindex(A::DocIdentityOperator{T}, i::Int, j::Int) where {T} =
    i == j ? one(T) : zero(T)

function LinearAlgebra.mul!(y::AbstractVector, ::DocIdentityOperator, x::AbstractVector)
    copyto!(y, x)
    return y
end

struct DocMatVecAlg <: LinearSolve.AbstractKrylovSubspaceMethod end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::DocMatVecAlg; kwargs...)
    mul!(cache.u, cache.A, cache.b)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = SciMLBase.ReturnCode.Success
    )
end

A = DocIdentityOperator{Float64}(3)
b = [1.0, 2.0, 3.0]
cache = init(LinearProblem(A, b), DocMatVecAlg())
@assert solve!(cache).u == b
```

The example is intentionally limited to the generic contract: a real iterative
algorithm must also define its convergence, stopping, and preconditioning rules
and should exercise those rules with dedicated tests.

## Checking compliance

```julia
using Test
@test isempty(LinearSolve.algorithm_interface_issues(MyLUFactorization()))
```

`algorithm_interface_issues` accepts either an instance or the type, and returns
one message per violation. Pass `check_solve = false` to check only the traits,
which is what LinearSolve.jl's own test suite does for algorithms whose `solve!`
lives in an extension whose backend is not loaded in that test group.

To sweep every algorithm a session knows about:

```julia
for T in LinearSolve.concrete_algorithm_types()
    @test isempty(LinearSolve.algorithm_interface_issues(T))
end
```

```@docs
LinearSolve.algorithm_interface_issues
LinearSolve.concrete_algorithm_types
LinearSolve.needs_square_A
LinearSolve.update_tolerances!
LinearSolve.update_tolerances_internal!
```
