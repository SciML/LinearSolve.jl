"""
`OperatorCondition`

Specifies the assumption of matrix conditioning for the default linear solver choices. Condition number
is defined as the ratio of eigenvalues. The numerical stability of many linear solver algorithms
can be dependent on the condition number of the matrix. The condition number can be computed as:

```julia
using LinearAlgebra
cond(rand(100, 100))
```

However, in practice this computation is very expensive and thus not possible for most practical cases.
Therefore, OperatorCondition lets one share to LinearSolve the expected conditioning. The higher the
expected condition number, the safer the algorithm needs to be and thus there is a trade-off between
numerical performance and stability. By default the method assumes the operator may be ill-conditioned
for the standard linear solvers to converge (such as LU-factorization), though more extreme
ill-conditioning or well-conditioning could be the case and specified through this assumption.
"""
EnumX.@enumx OperatorCondition begin
    """
    `OperatorCondition.IllConditioned`

    The default assumption of LinearSolve. Assumes that the operator can have minor ill-conditioning
    and thus needs to use safe algorithms.
    """
    IllConditioned
    """
    `OperatorCondition.VeryIllConditioned`

    Assumes that the operator can have fairly major ill-conditioning and thus the standard linear algebra
    algorithms cannot be used.
    """
    VeryIllConditioned
    """
    `OperatorCondition.SuperIllConditioned`

    Assumes that the operator can have fairly extreme ill-conditioning and thus the most stable algorithm
    is used.
    """
    SuperIllConditioned
    """
    `OperatorCondition.WellConditioned`

    Assumes that the operator can have fairly contained conditioning and thus the fastest algorithm is
    used.
    """
    WellConditioned
end

"""
    OperatorAssumptions(issquare = nothing; condition::OperatorCondition.T = IllConditioned)

Sets the operator `A` assumptions used as part of the default algorithm
"""
struct OperatorAssumptions{T}
    issq::T
    condition::OperatorCondition.T
end

function OperatorAssumptions(issquare = nothing;
        condition::OperatorCondition.T = OperatorCondition.IllConditioned)
    OperatorAssumptions{typeof(issquare)}(issquare, condition)
end
__issquare(assump::OperatorAssumptions) = assump.issq
__conditioning(assump::OperatorAssumptions) = assump.condition

"""
    LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr, Ttol, issq, S}

The core cache structure used by LinearSolve for storing and managing the state of linear
solver computations. This mutable struct acts as the primary interface for iterative 
solving and caching of factorizations and intermediate results.

## Fields

- `A::TA`: The matrix operator of the linear system.
- `b::Tb`: The right-hand side vector of the linear system.
- `u::Tu`: The solution vector (preallocated storage for the result).
- `p::Tp`: Parameters passed to the linear solver algorithm.
- `alg::Talg`: The linear solver algorithm instance.
- `cacheval::Tc`: Algorithm-specific cache storage for factorizations and intermediate computations.
- `isfresh::Bool`: Cache validity flag for the matrix `A`. `false` means `cacheval` is up-to-date 
  with respect to `A`, `true` means `cacheval` needs to be updated.
- `precsisfresh::Bool`: Cache validity flag for preconditioners. `false` means `Pl` and `Pr` 
  are up-to-date with respect to `A`, `true` means they need to be updated.
- `Pl::Tl`: Left preconditioner operator.
- `Pr::Tr`: Right preconditioner operator.
- `abstol::Ttol`: Absolute tolerance for iterative solvers.
- `reltol::Ttol`: Relative tolerance for iterative solvers.
- `maxiters::Int`: Maximum number of iterations for iterative solvers.
- `verbose::Bool`: Whether to print verbose output during solving.
- `assumptions::OperatorAssumptions{issq}`: Assumptions about the operator properties.
- `sensealg::S`: Sensitivity analysis algorithm for automatic differentiation.

## Usage

The `LinearCache` is typically created via `init(::LinearProblem, ::SciMLLinearSolveAlgorithm)` 
and then used with `solve!(cache)` for efficient repeated solves with the same matrix structure
but potentially different right-hand sides or parameter values.

## Cache Management

The cache automatically tracks when matrix `A` or parameters `p` change by setting the 
appropriate freshness flags. When `solve!` is called, stale cache entries are automatically
recomputed as needed.
"""
mutable struct LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr, Ttol, issq, S}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc  # store alg cache here
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    precsisfresh::Bool # false => PR,PL is set wrt A, true => update PR,PL wrt A
    Pl::Tl        # preconditioners
    Pr::Tr
    abstol::Ttol
    reltol::Ttol
    maxiters::Int
    verbose::LinearVerbosity
    assumptions::OperatorAssumptions{issq}
    sensealg::S
end

function Base.setproperty!(cache::LinearCache, name::Symbol, x)
    if name === :A
        setfield!(cache, :isfresh, true)
        setfield!(cache, :precsisfresh, true)
    elseif name === :p
        setfield!(cache, :precsisfresh, true)
    elseif name === :b
        # In case there is something that needs to be done when b is updated
        update_cacheval!(cache, :b, x)
    elseif name === :cacheval && cache.alg isa DefaultLinearSolver
        @assert cache.cacheval isa DefaultLinearSolverInit
        return __setfield!(cache.cacheval, cache.alg, x)
        # return setfield!(cache.cacheval, Symbol(cache.alg.alg), x)
    end
    setfield!(cache, name, x)
end

function update_cacheval!(cache::LinearCache, name::Symbol, x)
    return update_cacheval!(cache, cache.cacheval, name, x)
end
update_cacheval!(cache, cacheval, name::Symbol, x) = cacheval

"""
    init_cacheval(alg::SciMLLinearSolveAlgorithm, args...)

Initialize algorithm-specific cache values for the given linear solver algorithm.
This function returns `nothing` by default and is intended to be overloaded by 
specific algorithm implementations that need to store intermediate computations
or factorizations.

## Arguments
- `alg`: The linear solver algorithm instance
- `args...`: Additional arguments passed to the cache initialization

## Returns
Algorithm-specific cache value or `nothing` for algorithms that don't require caching.
"""
init_cacheval(alg::SciMLLinearSolveAlgorithm, args...) = nothing

function SciMLBase.init(prob::LinearProblem, args...; kwargs...)
    SciMLBase.init(prob, nothing, args...; kwargs...)
end

"""
    default_tol(T)

Compute the default tolerance for iterative linear solvers based on the element type.
The tolerance is typically set as the square root of the machine epsilon for the 
given floating point type, ensuring numerical accuracy appropriate for that precision.

## Arguments
- `T`: The element type of the linear system

## Returns
- For floating point types: `√(eps(T))`
- For exact types (Rational, Integer): `0` (exact arithmetic)
- For Any type: `0` (conservative default)
"""
default_tol(::Type{T}) where {T} = √(eps(T))
default_tol(::Type{Complex{T}}) where {T} = √(eps(T))
default_tol(::Type{<:Rational}) = 0
default_tol(::Type{<:Integer}) = 0
default_tol(::Type{Any}) = 0

"""
    default_alias_A(alg, A, b) -> Bool

Determine the default aliasing behavior for the matrix `A` given the algorithm type.
Aliasing allows the algorithm to modify the original matrix in-place for efficiency,
but this may not be desirable or safe for all algorithm types.

## Arguments
- `alg`: The linear solver algorithm
- `A`: The matrix operator  
- `b`: The right-hand side vector

## Returns
- `false`: Safe default, algorithm will not modify the original matrix `A`
- `true`: Algorithm may modify `A` in-place for efficiency

## Algorithm-Specific Behavior
- Dense factorizations: `false` (destructive, need to preserve original)
- Krylov methods: `true` (non-destructive, safe to alias)
- Sparse factorizations: `true` (typically preserve sparsity structure)
"""
default_alias_A(::Any, ::Any, ::Any) = false

"""
    default_alias_b(alg, A, b) -> Bool

Determine the default aliasing behavior for the right-hand side vector `b` given the 
algorithm type. Similar to `default_alias_A` but for the RHS vector.

## Returns
- `false`: Safe default, algorithm will not modify the original vector `b`
- `true`: Algorithm may modify `b` in-place for efficiency
"""
default_alias_b(::Any, ::Any, ::Any) = false

# Non-destructive algorithms default to true
default_alias_A(::AbstractKrylovSubspaceMethod, ::Any, ::Any) = true
default_alias_b(::AbstractKrylovSubspaceMethod, ::Any, ::Any) = true

default_alias_A(::AbstractSparseFactorization, ::Any, ::Any) = true
default_alias_b(::AbstractSparseFactorization, ::Any, ::Any) = true

DEFAULT_PRECS(A, p) = IdentityOperator(size(A)[1]), IdentityOperator(size(A)[2])

"""
    __init_u0_from_Ab(A, b)

Initialize the solution vector `u0` with appropriate size and type based on the 
matrix `A` and right-hand side `b`. The solution vector is allocated with the 
same element type as `b` and sized to match the number of columns in `A`.

## Arguments
- `A`: The matrix operator (determines solution vector size)
- `b`: The right-hand side vector (determines element type)

## Returns
A zero-initialized vector of size `(size(A, 2),)` with element type matching `b`.

## Specializations
- For static matrices (`SMatrix`): Returns a static vector (`SVector`)
- For regular matrices: Returns a similar vector to `b` with appropriate size
"""
function __init_u0_from_Ab(A, b)
    u0 = similar(b, size(A, 2))
    fill!(u0, false)
    return u0
end
__init_u0_from_Ab(::SMatrix{S1, S2}, b) where {S1, S2} = zeros(SVector{S2, eltype(b)})

function SciMLBase.init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    __init(prob, alg, args...; kwargs...)
end

function __init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        abstol = default_tol(real(eltype(prob.b))),
        reltol = default_tol(real(eltype(prob.b))),
        maxiters::Int = length(prob.b),
        verbose::LinearVerbosity = false,
        Pl = nothing,
        Pr = nothing,
        assumptions = OperatorAssumptions(issquare(prob.A)),
        sensealg = LinearSolveAdjoint(),
        kwargs...)
    (; A, b, u0, p) = prob

    if haskey(kwargs, :alias_A) || haskey(kwargs, :alias_b)
        aliases = LinearAliasSpecifier()

        if haskey(kwargs, :alias_A)
            message = "`alias_A` keyword argument is deprecated, to set `alias_A`,
            please use an LinearAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_A = true))"
            Base.depwarn(message, :init)
            Base.depwarn(message, :solve)
            aliases = LinearAliasSpecifier(alias_A = values(kwargs).alias_A)
        end

        if haskey(kwargs, :alias_b)
            message = "`alias_b` keyword argument is deprecated, to set `alias_b`,
            please use an LinearAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_b = true))"
            Base.depwarn(message, :init)
            Base.depwarn(message, :solve)
            aliases = LinearAliasSpecifier(
                alias_A = aliases.alias_A, alias_b = values(kwargs).alias_b)
        end
    else
        if alias isa Bool
            aliases = LinearAliasSpecifier(alias = alias)
        else
            aliases = alias
        end
    end

    if isnothing(aliases.alias_A)
        alias_A = default_alias_A(alg, prob.A, prob.b)
    else
        alias_A = aliases.alias_A
    end

    if isnothing(aliases.alias_b)
        alias_b = default_alias_b(alg, prob.A, prob.b)
    else
        alias_b = aliases.alias_b
    end

    A = if alias_A || A isa SMatrix
        A
    elseif A isa Array
        copy(A)
    elseif issparsematrixcsc(A)
        make_SparseMatrixCSC(A)
    else
        deepcopy(A)
    end

    b = if issparsematrix(b) && !(A isa Diagonal)
        Array(b) # the solution to a linear solve will always be dense!
    elseif alias_b || b isa SVector
        b
    elseif b isa Array
        copy(b)
    elseif issparsematrixcsc(b)
        # Extension must be loaded if issparsematrixcsc returns true
        make_SparseMatrixCSC(b)
    else
        deepcopy(b)
    end

    u0_ = u0 !== nothing ? u0 : __init_u0_from_Ab(A, b)

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(eltype(prob.b))(reltol)
    abstol = real(eltype(prob.b))(abstol)

    precs = if hasproperty(alg, :precs)
        isnothing(alg.precs) ? DEFAULT_PRECS : alg.precs
    else
        DEFAULT_PRECS
    end
    _Pl, _Pr = precs(A, p)
    if isnothing(Pl)
        Pl = _Pl
    else
        # TODO: deprecate once all docs are updated to the new form
        #@warn "passing Preconditioners at `init`/`solve` time is deprecated. Instead add a `precs` function to your algorithm."
    end
    if isnothing(Pr)
        Pr = _Pr
    else
        # TODO: deprecate once all docs are updated to the new form
        #@warn "passing Preconditioners at `init`/`solve` time is deprecated. Instead add a `precs` function to your algorithm."
    end
    cacheval = init_cacheval(alg, A, b, u0_, Pl, Pr, maxiters, abstol, reltol, verbose,
        assumptions)
    isfresh = true
    precsisfresh = false
    Tc = typeof(cacheval)

    cache = LinearCache{typeof(A), typeof(b), typeof(u0_), typeof(p), typeof(alg), Tc,
        typeof(Pl), typeof(Pr), typeof(reltol), typeof(assumptions.issq),
        typeof(sensealg)}(
        A, b, u0_, p, alg, cacheval, isfresh, precsisfresh, Pl, Pr, abstol, reltol,
        maxiters, verbose, assumptions, sensealg)
    return cache
end

function SciMLBase.reinit!(cache::LinearCache;
        A = nothing,
        b = cache.b,
        u = cache.u,
        p = nothing,
        reuse_precs = false)
    (; alg, cacheval, abstol, reltol, maxiters, verbose, assumptions, sensealg) = cache

    isfresh = !isnothing(A)
    precsisfresh = !reuse_precs && (isfresh || !isnothing(p))
    isfresh |= cache.isfresh
    precsisfresh |= cache.precsisfresh

    A = isnothing(A) ? cache.A : A
    b = isnothing(b) ? cache.b : b
    u = isnothing(u) ? cache.u : u
    p = isnothing(p) ? cache.p : p
    Pl = cache.Pl
    Pr = cache.Pr

    cache.A = A
    cache.b = b
    cache.u = u
    cache.p = p
    cache.Pl = Pl
    cache.Pr = Pr
    cache.isfresh = true
    cache.precsisfresh = precsisfresh
    nothing
end

function SciMLBase.solve(prob::LinearProblem, args...; kwargs...)
    return solve(prob, nothing, args...; kwargs...)
end

function SciMLBase.solve(prob::LinearProblem, ::Nothing, args...;
        assump = OperatorAssumptions(issquare(prob.A)), kwargs...)
    return solve(prob, defaultalg(prob.A, prob.b, assump), args...; kwargs...)
end

function SciMLBase.solve(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...; kwargs...)
    solve!(init(prob, alg, args...; kwargs...))
end

function SciMLBase.solve!(cache::LinearCache, args...; kwargs...)
    solve!(cache, cache.alg, args...; kwargs...)
end

# Special Case for StaticArrays
const StaticLinearProblem = LinearProblem{uType, iip, <:SMatrix,
    <:Union{<:SMatrix, <:SVector}} where {uType, iip}

function SciMLBase.solve(prob::StaticLinearProblem, args...; kwargs...)
    return SciMLBase.solve(prob, nothing, args...; kwargs...)
end

function SciMLBase.solve(prob::StaticLinearProblem,
        alg::Nothing, args...; kwargs...)
    u = prob.A \ prob.b
    return SciMLBase.build_linear_solution(
        alg, u, nothing, prob; retcode = ReturnCode.Success)
end

function SciMLBase.solve(prob::StaticLinearProblem,
        alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    if alg === nothing || alg isa DirectLdiv!
        u = prob.A \ prob.b
    elseif alg isa LUFactorization
        u = lu(prob.A) \ prob.b
    elseif alg isa QRFactorization
        u = qr(prob.A) \ prob.b
    elseif alg isa CholeskyFactorization
        u = cholesky(prob.A) \ prob.b
    elseif alg isa NormalCholeskyFactorization
        u = cholesky(Symmetric(prob.A' * prob.A)) \ (prob.A' * prob.b)
    elseif alg isa SVDFactorization
        u = svd(prob.A) \ prob.b
    else
        # Slower Path but handles all cases
        cache = init(prob, alg, args...; kwargs...)
        return solve!(cache)
    end
    return SciMLBase.build_linear_solution(
        alg, u, nothing, prob; retcode = ReturnCode.Success)
end
