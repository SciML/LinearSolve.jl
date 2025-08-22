# Developing New Linear Solvers

Developing new or custom linear solvers for the SciML interface can be done in
one of two ways:

 1. You can either create a completely new set of dispatches for `init` and `solve`.
 2. You can extend LinearSolve.jl's internal mechanisms.

For developer ease, we highly recommend (2) as that will automatically make the
caching API work. Thus, this is the documentation for how to do that.

## Developing New Linear Solvers with LinearSolve.jl Primitives

Let's create a new wrapper for a simple LU-factorization which uses only the
basic machinery. A simplified version is:

```julia
struct MyLUFactorization{P} <: LinearSolve.SciMLLinearSolveAlgorithm end

function LinearSolve.init_cacheval(
        alg::MyLUFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Bool, assump::LinearSolve.OperatorAssumptions)
    lu!(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::MyLUFactorization; kwargs...)
    if cache.isfresh
        A = cache.A
        A = convert(AbstractMatrix, A)
        fact = lu!(A)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, cache.cacheval, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end
```

The way this works is as follows. LinearSolve.jl has a `LinearCache` that everything
shares (this is what gives most of the ease of use). However, many algorithms
need to cache their own things, and so there's one value `cacheval` that is
for the algorithms to modify. The function:

```julia
init_cacheval(
    alg::MyLUFactorization, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose, assump)
```

is what is called at `init` time to create the first `cacheval`. Note that this
should match the type of the cache later used in `solve` as many algorithms, like
those in OrdinaryDiffEq.jl, expect type-groundedness in the linear solver definitions.
While there are cheaper ways to obtain this type for LU factorizations (specifically,
`ArrayInterface.lu_instance(A)`), for a demonstration, this just performs an
LU-factorization to get an `LU{T, Matrix{T}}` which it puts into the `cacheval`
so it is typed for future use.

After the `init_cacheval`, the only thing left to do is to define
`SciMLBase.solve!(cache::LinearCache, alg::MyLUFactorization)`. Many algorithms
may use a lazy matrix-free representation of the operator `A`. Thus, if the
algorithm requires a concrete matrix, like LU-factorization does, the algorithm
should `convert(AbstractMatrix,cache.A)`. The flag `cache.isfresh` states whether
`A` has changed since the last `solve`. Since we only need to factorize when
`A` is new, the factorization part of the algorithm is done in a `if cache.isfresh`.
`cache.cacheval = fact; cache.isfresh = false` puts the new factorization into the cache,
so it's updated for future solves. Then `y = ldiv!(cache.u, cache.cacheval, cache.b)`
performs the solve and a linear solution is returned via
`SciMLBase.build_linear_solution(alg,y,nothing,cache)`.
