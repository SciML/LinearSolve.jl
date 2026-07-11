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
    EnumX.@enumx NonstructuralZeros

How a sparse operator's *nonstructural zeros* — stored entries that are
numerically zero — are expected to behave across a sequence of solves. Such
stored zeros (common in ODE/DAE Jacobians and `W = I - γJ` built from a
conservative symbolic sparsity pattern) join the fill-reducing ordering and
symbolic factorization as if real, inflating the factor, so dropping them speeds
up every refactor and solve. Passed via [`OperatorAssumptions`](@ref); has no
effect on dense operators.
"""
EnumX.@enumx NonstructuralZeros begin
    """
    `NonstructuralZeros.Auto`

    Default. Detect from the starting matrix: enable the reduction when a
    sufficient fraction of the stored entries are numerically zero (see
    `LinearSolve.PERSISTENT_ZERO_FRACTION_THRESHOLD`), starting in cached-union
    mode and switching to per-solve `dropzeros` if the zeros prove non-persistent
    (more than `LinearSolve.NONPERSISTENT_ZERO_FRACTION` of the starting zeros
    activate).
    """
    Auto
    """
    `NonstructuralZeros.None`

    Assume the operator has no nonstructural zeros worth dropping. Never reduce —
    bit-for-bit identical to the plain factorization, with no detection overhead.
    """
    None
    """
    `NonstructuralZeros.Persistent`

    Assume nonstructural zeros are present at *persistent* positions (the same
    entries stay zero across solves). Drop them via the cached union of
    ever-nonzero positions, reusing the symbolic factorization across solves.
    """
    Persistent
    """
    `NonstructuralZeros.Present`

    Assume nonstructural zeros are present but at positions that may vary between
    solves. Drop each matrix's own zeros per solve (no cross-solve symbolic
    caching; the inner solver re-analyzes when the pattern changes).
    """
    Present
end

"""
    OperatorAssumptions(issquare = nothing;
                        condition::OperatorCondition.T = IllConditioned,
                        nonstructural_zeros::NonstructuralZeros.T = Auto)

Sets the operator `A` assumptions used as part of the default algorithm.

`issquare` asserts whether `A` is square (and thus whether a direct
factorization vs. a least-squares solver is appropriate). `nothing` (default)
defers the decision, letting `init`/`defaultalg` infer it from `A`.

`condition` describes the conditioning of `A` and selects how aggressively the
default algorithm trades speed for stability (see [`OperatorCondition`](@ref)):

  - `OperatorCondition.IllConditioned` (default): assume `A` may be ill
    conditioned; pick a stability-preserving algorithm (e.g. pivoted
    factorizations).
  - `OperatorCondition.WellConditioned`: assume contained conditioning and pick
    the fastest algorithm, skipping safety work.
  - `OperatorCondition.VeryIllConditioned` /
    `OperatorCondition.SuperIllConditioned`: progressively more conservative,
    favoring the most numerically robust paths.

`nonstructural_zeros` declares how `A`'s *nonstructural zeros* (stored entries
that are numerically zero) behave across a sequence of solves, and hence whether
and how a sparse factorization should drop them (see [`NonstructuralZeros`](@ref)):

  - `NonstructuralZeros.Auto` (default): detect from the starting matrix and
    adapt (cached union, falling back to per-solve dropzeros if non-persistent).
  - `NonstructuralZeros.None`: none worth dropping; never reduce (bit-identical).
  - `NonstructuralZeros.Persistent`: present at stable positions; cached-union
    reduction.
  - `NonstructuralZeros.Present`: present but positions may vary; per-solve
    dropzeros.

Has no effect on dense `A`.
"""
struct OperatorAssumptions{T}
    issq::T
    condition::OperatorCondition.T
    nonstructural_zeros::NonstructuralZeros.T
end

function OperatorAssumptions(
        issquare = nothing;
        condition::OperatorCondition.T = OperatorCondition.IllConditioned,
        nonstructural_zeros::NonstructuralZeros.T = NonstructuralZeros.Auto
    )
    return OperatorAssumptions{typeof(issquare)}(
        issquare, condition, nonstructural_zeros
    )
end
__issquare(assump::OperatorAssumptions) = assump.issq
__conditioning(assump::OperatorAssumptions) = assump.condition
__nonstructural_zeros(assump::OperatorAssumptions) = assump.nonstructural_zeros

# Fraction of stored entries that must be numerically zero on the *starting*
# matrix for auto-detection (`nonstructural_zeros == NonstructuralZeros.Auto`) to
# enable the sparse reduction. Below this the matrix is treated as already tight
# and factorized unchanged (no detection overhead, bit-identical).
const PERSISTENT_ZERO_FRACTION_THRESHOLD = 0.1

# In auto mode, if more than this fraction of the entries that were numerically
# zero on the *starting* matrix have since become nonzero, the nonstructural zeros
# are deemed non-persistent (they wobble too much for a stable reduced pattern).
# The reduction then stops maintaining the union and instead drops each matrix's
# own zeros per solve (no cross-solve symbolic caching) — better than carrying a
# union that has lost most of what it could drop. `NonstructuralZeros.Persistent`
# pins union caching and never switches; `NonstructuralZeros.Present` starts in
# per-solve mode. (This is "fraction of the starting zeros that turned nonzero",
# independent of how dense the matrix is — not a fraction of the whole stored
# pattern.)
const NONPERSISTENT_ZERO_FRACTION = 0.5

# Shared persistent-nonstructural-zero reduction helpers. The reduction drops
# stored entries that have been numerically zero in every solve so far (the
# complement of the running union of ever-nonzero positions), handing the inner
# factorization a smaller, valid superset pattern. Sparse-matrix methods live in
# the SparseArrays extension; these generic fallbacks make the non-sparse /
# reduction-off paths no-ops so callers can stay branch-light and type-stable.
#
# `init_sparse_reduction(A, assumptions)` returns either `nothing` (no reduction
# for this A) or a concrete reduction-state object; `reduce_operand!(red, A)`
# returns the matrix to factor (the reduced operand when active, else `A`).
init_sparse_reduction(A, assumptions) = nothing
reduce_operand!(::Nothing, A) = A

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
- `verbose::LinearVerbosity`: Whether to print verbose output during solving.
- `assumptions::OperatorAssumptions{issq}`: Assumptions about the operator properties.
- `sensealg::S`: Sensitivity analysis algorithm for automatic differentiation.
- `alias_A::Bool`: The resolved `LinearAliasSpecifier.alias_A` from `init`. When `true`,
  the user has permitted LinearSolve to overwrite `A`; dense factorizations may then
  refactorize in place (e.g. `lu!(cache.A)`) after `cache.A` is replaced, skipping the
  O(n²) copy.

## Usage

The `LinearCache` is typically created via `init(::LinearProblem, ::SciMLLinearSolveAlgorithm)` 
and then used with `solve!(cache)` for efficient repeated solves with the same matrix structure
but potentially different right-hand sides or parameter values.

## Cache Management

The cache automatically tracks when matrix `A` or parameters `p` change by setting the 
appropriate freshness flags. When `solve!` is called, stale cache entries are automatically
recomputed as needed.
"""
mutable struct LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr, Ttol, Tlv <: LinearVerbosity, issq, S, Tred}
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
    verbose::Tlv
    assumptions::OperatorAssumptions{issq}
    sensealg::S
    # Persistent-nonstructural-zero reduction state for standalone sparse
    # factorizations (`nothing` otherwise; the default solver carries its own in
    # `DefaultLinearSolverInit`). Set once at `init`; persists across `reinit!`.
    sparse_reduction::Tred
    # Resolved `LinearAliasSpecifier.alias_A` from `init` (defaults applied, so
    # never `nothing`). `true` means the user permitted overwriting `A`, which
    # also permits in-place refactorization (e.g. `lu!(A)`) after `cache.A = X`.
    alias_A::Bool
end

function Base.setproperty!(cache::LinearCache, name::Symbol, x)
    if name === :A
        setfield!(cache, :isfresh, true)
        setfield!(cache, :precsisfresh, true)
        if cache.cacheval isa DefaultLinearSolverInit
            cache.cacheval.fell_back_to_qr = false
            if x === getfield(cache, :A) && cache.cacheval.a_backup_allocated
                A_backup = cache.cacheval.A_backup
                if size(A_backup) == size(x)
                    copyto!(A_backup, x)
                else
                    setfield!(cache.cacheval, :A_backup, copy(x))
                end
                cache.cacheval.a_backup_synced = true
            elseif !(x === getfield(cache, :A))
                # A was replaced by a different object; A_backup is now stale
                cache.cacheval.a_backup_synced = false
            end
        end
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
    return setfield!(cache, name, x)
end

function Base.resize!(cache::LinearCache, i::Int)
    resize_cacheval!(cache, cache.cacheval, i)
    setfield!(cache, :isfresh, true)
    return cache
end

resize_cacheval!(cache, cacheval, i) = nothing

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
    return SciMLBase.init(prob, nothing, args...; kwargs...)
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

# Default verbose setting (const for type stability)
const DEFAULT_VERBOSE = LinearVerbosity()

# Helper functions for processing verbose parameter with multiple dispatch (type-stable)
@inline _process_verbose_param(verbose::LinearVerbosity) = (verbose, verbose)
@inline function _process_verbose_param(verbose::SciMLLogging.AbstractVerbosityPreset)
    verbose_spec = LinearVerbosity(verbose)
    return (verbose_spec, verbose_spec)
end
@inline function _process_verbose_param(verbose::Bool)
    # @warn "Using `true` or `false` for `verbose` is being deprecated."
    verbose_spec = verbose ? DEFAULT_VERBOSE : LinearVerbosity(SciMLLogging.None())
    return (verbose_spec, verbose)
end

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
For a matrix (batched) right-hand side `b` of size `(size(A, 1), k)`, returns a
zero-initialized matrix of size `(size(A, 2), k)` so that each column of `u0`
corresponds to a column of `b`.

## Specializations
- For static matrices (`SMatrix`): Returns a static vector (`SVector`)
- For regular matrices: Returns a similar vector to `b` with appropriate size
"""
function __init_u0_from_Ab(A, b)
    u0 = similar(b, size(A, 2))
    fill!(u0, false)
    return u0
end
function __init_u0_from_Ab(A, b::AbstractMatrix)
    u0 = similar(b, size(A, 2), size(b, 2))
    fill!(u0, false)
    return u0
end
__init_u0_from_Ab(::SMatrix{S1, S2}, b) where {S1, S2} = zeros(SVector{S2, eltype(b)})
function __init_u0_from_Ab(::SMatrix{S1, S2}, b::AbstractMatrix) where {S1, S2}
    u0 = similar(b, S2, size(b, 2))
    fill!(u0, false)
    return u0
end
function __init_u0_from_Ab(
        ::SMatrix{S1, S2}, ::SMatrix{S1b, S2b, Tb}
    ) where {S1, S2, S1b, S2b, Tb}
    return zeros(SMatrix{S2, S2b, Tb})
end

"""
    _check_batched_rhs_support(alg, b)

Throw an informative `ArgumentError` at `init` time when a matrix (batched)
right-hand side `b` is used with an algorithm that only supports vector `b`
(Krylov subspace / iterative methods). Factorization-based algorithms support
matrix `b` and pass through the generic no-op fallback.
"""
_check_batched_rhs_support(alg, b) = nothing
function _check_batched_rhs_support(alg::AbstractKrylovSubspaceMethod, b::AbstractMatrix)
    throw(
        ArgumentError(
            "Batched (matrix) right-hand sides are only supported by factorization " *
                "algorithms and block Krylov methods; $(nameof(typeof(alg))) supports " *
                "only vector `b`. Use KrylovJL_GMRES/KrylovJL_MINRES (block methods), " *
                "a factorization algorithm (e.g. `LUFactorization()`), or solve " *
                "column-by-column."
        )
    )
end
function _check_batched_rhs_support(alg::DefaultLinearSolver, b::AbstractMatrix)
    # KrylovJL_GMRES is fine: it dispatches to Krylov.jl's block GMRES for
    # matrix b. CRAIGMR/LSMR (least-squares operator defaults) have no block
    # variants.
    if alg.alg === DefaultAlgorithmChoice.KrylovJL_CRAIGMR ||
            alg.alg === DefaultAlgorithmChoice.KrylovJL_LSMR
        throw(
            ArgumentError(
                "Batched (matrix) right-hand sides are not supported by the " *
                    "least-squares Krylov method $(alg.alg) the default algorithm " *
                    "selected for this operator. Solve column-by-column or use a " *
                    "factorization algorithm."
            )
        )
    end
    return nothing
end

function SciMLBase.init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    return __init(prob, alg, args...; kwargs...)
end

function __init(
        prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        abstol = default_tol(real(eltype(prob.b))),
        reltol = default_tol(real(eltype(prob.b))),
        maxiters::Int = length(prob.b),
        verbose = LinearVerbosity(),
        Pl = nothing,
        Pr = nothing,
        assumptions = OperatorAssumptions(issquare(prob.A)),
        sensealg = LinearSolveAdjoint(),
        kwargs...
    )
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
                alias_A = aliases.alias_A, alias_b = values(kwargs).alias_b
            )
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
    elseif A isa Adjoint
        adjoint(copy(parent(A)))
    elseif A isa Transpose
        transpose(copy(parent(A)))
    else
        copy(A)
    end

    verbose_spec, init_cache_verb = _process_verbose_param(verbose)

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
        copy(b)
    end

    _check_batched_rhs_support(alg, b)

    u0_ = u0 !== nothing ? u0 : __init_u0_from_Ab(A, b)

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = real(eltype(prob.b))(SciMLBase.value(reltol))
    abstol = real(eltype(prob.b))(SciMLBase.value(abstol))

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
    # For DefaultLinearSolver, pass original prob.A so the A_backup field gets the
    # correct type at construction time (prob.A may be e.g. WOperator while the
    # converted A used for sub-caches may be a different concrete type).
    cacheval = if alg isa DefaultLinearSolver
        init_cacheval(
            alg, A, b, u0_, Pl, Pr, maxiters, abstol, reltol, init_cache_verb,
            assumptions, prob.A
        )
    else
        init_cacheval(
            alg, A, b, u0_, Pl, Pr, maxiters, abstol, reltol, init_cache_verb,
            assumptions
        )
    end
    isfresh = true
    precsisfresh = false
    Tc = typeof(cacheval)

    # Standalone sparse factorizations may drop persistent nonstructural zeros (the
    # default carries its own reduction in DefaultLinearSolverInit, so skip it here).
    sparse_reduction = alg isa AbstractSparseFactorization ?
        init_sparse_reduction(A, assumptions) : nothing

    cache = LinearCache{
        typeof(A), typeof(b), typeof(u0_), typeof(p), typeof(alg), Tc,
        typeof(Pl), typeof(Pr), typeof(reltol), typeof(verbose_spec), typeof(assumptions.issq),
        typeof(sensealg), typeof(sparse_reduction),
    }(
        A, b, u0_, p, alg, cacheval, isfresh, precsisfresh, Pl, Pr, abstol, reltol,
        maxiters, verbose_spec, assumptions, sensealg, sparse_reduction, alias_A
    )
    return cache
end

function SciMLBase.reinit!(
        cache::LinearCache;
        A = nothing,
        b = cache.b,
        u = cache.u,
        p = nothing,
        reuse_precs = false
    )
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
    cache.isfresh = isfresh
    cache.precsisfresh = precsisfresh
    return nothing
end

function SciMLBase.solve(prob::LinearProblem, args...; kwargs...)
    return solve(prob, nothing, args...; kwargs...)
end

function SciMLBase.solve(
        prob::LinearProblem, ::Nothing, args...;
        assump = OperatorAssumptions(issquare(prob.A)), kwargs...
    )
    return solve(prob, defaultalg(prob.A, prob.b, assump), args...; kwargs...)
end

function SciMLBase.solve(
        prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...; kwargs...
    )
    return solve!(init(prob, alg, args...; kwargs...))
end

function SciMLBase.solve!(cache::LinearCache, args...; kwargs...)
    return solve!(cache, cache.alg, args...; kwargs...)
end

# Special Case for StaticArrays
const StaticLinearProblem = LinearProblem{
    uType, iip, <:SMatrix,
    <:Union{<:SMatrix, <:SVector},
} where {uType, iip}

function SciMLBase.solve(prob::StaticLinearProblem, args...; kwargs...)
    return SciMLBase.solve(prob, nothing, args...; kwargs...)
end

function SciMLBase.solve(
        prob::StaticLinearProblem,
        alg::Nothing, args...; kwargs...
    )
    u = prob.A \ prob.b
    return SciMLBase.build_linear_solution(
        alg, u, nothing, nothing; retcode = ReturnCode.Success
    )
end

function SciMLBase.solve(
        prob::StaticLinearProblem,
        alg::SciMLLinearSolveAlgorithm, args...; kwargs...
    )
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
        alg, u, nothing, nothing; retcode = ReturnCode.Success
    )
end

function update_tolerances!(cache; abstol = nothing, reltol = nothing)
    if abstol !== nothing
        cache.abstol = abstol
    end
    if reltol !== nothing
        cache.reltol = reltol
    end
    return update_tolerances_internal!(cache, cache.alg, abstol, reltol)
end


function update_tolerances_internal!(cache, alg::AbstractFactorization, abstol, reltol)
    error("Cannot update tolerances for factorization.")
end

function update_tolerances_internal!(cache, alg::AbstractKrylovSubspaceMethod, abstol, reltol)
    return @warn "Tolerance update for Krylov subspace method '$typeof(alg)' not implemented." maxlog = 1
end
