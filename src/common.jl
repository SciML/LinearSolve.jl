"""
    EnumX.@enumx OperatorCondition

Specifies the assumption of matrix conditioning for the default linear solver choices.
The condition number is the ratio of the largest to the smallest singular value of `A`
(for normal matrices this coincides with the ratio of the extreme eigenvalue magnitudes).
The numerical stability of many linear solver algorithms can be dependent on the
condition number of the matrix. The condition number can be computed as:

```julia
using LinearAlgebra
cond(rand(100, 100))
```

However, in practice this computation is very expensive and thus not possible for most
practical cases. Therefore, `OperatorCondition` lets one share to LinearSolve the
expected conditioning. The higher the expected condition number, the safer the
algorithm needs to be and thus there is a trade-off between numerical performance and
stability. By default the method assumes the operator may be ill-conditioned for the
standard linear solvers to converge (such as LU-factorization), though more extreme
ill-conditioning or well-conditioning could be the case and specified through this
assumption.

The assumption is supplied through the `condition` keyword of `OperatorAssumptions`,
which in turn is passed as the `assumptions` keyword of `init` (or the `assump`
keyword of `solve` when no algorithm is given):

```julia
cache = init(prob; assumptions = OperatorAssumptions(true;
    condition = OperatorCondition.WellConditioned))
sol = solve!(cache)
```

It only affects the default algorithm (`solve(prob)` with no algorithm given). The
members and their effect on the dense default choice are:

  - `OperatorCondition.IllConditioned` (default): pivoted LU for square `A`
    (`LUFactorization` or one of its size- and BLAS-dependent variants), QR
    (column-pivoted when underdetermined) for non-square `A`.
  - `OperatorCondition.WellConditioned`: LU for square `A` (as above),
    `NormalCholeskyFactorization` for non-square `A`.
  - `OperatorCondition.VeryIllConditioned`: `QRFactorization` (column-pivoted when
    underdetermined).
  - `OperatorCondition.SuperIllConditioned`: `SVDFactorization`.

Sparse and structured `A` have their own default choices that do not consult this
setting; on GPU arrays `IllConditioned` (or a non-square `A`) selects QR and the other
members select LU.
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
                        condition::OperatorCondition.T = OperatorCondition.IllConditioned,
                        nonstructural_zeros::NonstructuralZeros.T = NonstructuralZeros.Auto)

Sets the operator `A` assumptions used as part of the default algorithm. The object is
supplied through the `assumptions` keyword of `init` (or the `assump` keyword of
`solve` when no algorithm is given), whose default is
`OperatorAssumptions(issquare(prob.A))`:

```julia
cache = init(prob; assumptions = OperatorAssumptions(true;
    condition = OperatorCondition.WellConditioned))
sol = solve!(cache)
```

## Positional Arguments

  - `issquare`: asserts whether `A` is square (and thus whether a direct
    factorization vs. a least-squares solver is appropriate). `nothing` (default)
    defers the decision, letting `init`/`defaultalg` infer it from `A`.

## Keyword Arguments

  - `condition`: describes the conditioning of `A` and selects how aggressively the
    default algorithm trades speed for stability (see `OperatorCondition`). Defaults
    to `OperatorCondition.IllConditioned`.

      + `OperatorCondition.IllConditioned` (default): assume `A` may be ill
        conditioned; pick a stability-preserving algorithm (e.g. pivoted
        factorizations).
      + `OperatorCondition.WellConditioned`: assume contained conditioning and pick
        the fastest algorithm, skipping safety work.
      + `OperatorCondition.VeryIllConditioned` /
        `OperatorCondition.SuperIllConditioned`: progressively more conservative,
        favoring the most numerically robust paths.

  - `nonstructural_zeros`: declares how `A`'s *nonstructural zeros* (stored entries
    that are numerically zero) behave across a sequence of solves, and hence whether
    and how a sparse factorization should drop them (see `NonstructuralZeros`).
    Defaults to `NonstructuralZeros.Auto`.

      + `NonstructuralZeros.Auto` (default): detect from the starting matrix and
        adapt (cached union, falling back to per-solve dropzeros if non-persistent).
      + `NonstructuralZeros.None`: none worth dropping; never reduce (bit-identical).
      + `NonstructuralZeros.Persistent`: present at stable positions; cached-union
        reduction.
      + `NonstructuralZeros.Present`: present but positions may vary; per-solve
        dropzeros.

`issquare` and `condition` steer the dense default choice (LU vs. QR vs. SVD vs.
`NormalCholeskyFactorization`); `nonstructural_zeros` only applies to sparse `A` and
has no effect on dense `A`. Sparse and structured `A` consult at most `issquare` when
picking their default.
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
# `src/sparsearrays.jl`; these generic fallbacks make the non-sparse /
# reduction-off paths no-ops so callers can stay branch-light and type-stable.
#
# `init_sparse_reduction(A, assumptions)` returns either `nothing` (no reduction
# for this A) or a concrete reduction-state object; `reduce_operand!(red, A)`
# returns the matrix to factor (the reduced operand when active, else `A`).
init_sparse_reduction(A, assumptions) = nothing
reduce_operand!(::Nothing, A) = A

"""
    LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr, Ttol, issq, S}

The mutable state passed to a linear solver algorithm by `init` and reused by
`solve!`. Construct it with `SciMLBase.init(::LinearProblem, alg)` rather than
calling the constructor directly.

# Fields

  - `A::TA`: Operator or matrix for the system.
  - `b::Tb`: Right-hand side. It may be a vector or a matrix of right-hand sides.
  - `u::Tu`: Preallocated solution storage written by `solve!`.
  - `p::Tp`: Problem parameters forwarded to the algorithm.
  - `alg::Talg`: Algorithm instance used by this cache.
  - `cacheval::Tc`: Algorithm-owned factorization, workspace, or solver object.
  - `isfresh::Bool`: Whether `cacheval` must be rebuilt because `A` changed.
  - `precsisfresh::Bool`: Whether the preconditioners must be refreshed.
  - `Pl::Tl`: Left preconditioner, or `nothing`.
  - `Pr::Tr`: Right preconditioner, or `nothing`.
  - `abstol::Ttol`: Absolute convergence tolerance.
  - `reltol::Ttol`: Relative convergence tolerance.
  - `maxiters::Int`: Maximum iteration count for iterative algorithms.
  - `verbose::Tlv`: Verbosity specification.
  - `assumptions::OperatorAssumptions{issq}`: Properties promised about `A`.
  - `sensealg::S`: Sensitivity algorithm associated with the solve.
  - `sparse_reduction::Tred`: State for persistent sparse-pattern reduction, or `nothing`.
  - `alias_A::Bool`: Whether the caller permits replacing or mutating `A`.

# Interface rules

An algorithm's `solve!` method may update `u`, `cacheval`, and the freshness
flags, but must preserve the meaning of the other fields. When `isfresh` is
`true`, rebuild any factorization or backend object that depends on `A`; after
doing so, set it to `false`. Algorithms that read tolerances at solve time use
`abstol`, `reltol`, and `maxiters` directly. Algorithms that copy tolerances into
`cacheval` must implement `update_tolerances_internal!`.

# Examples

```julia
prob = LinearProblem(A, b)
cache = init(prob, LUFactorization())
sol = solve!(cache)
cache.b = b2
sol2 = solve!(cache)
```
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

# `@inline` is load-bearing: `name` must constant-propagate into the body so the
# branch chain and `fieldtype` fold away. Without it, the warm refactorization
# loop's `cache.isfresh = false`-style assignments inside `solve!` degrade to
# dynamic calls and allocate, breaking the 0-byte ceilings in
# test/Core/lu_refactorization.jl on the default-algorithm paths.
@inline function Base.setproperty!(cache::LinearCache, name::Symbol, x)
    # Default `setproperty!` semantics convert to the field's declared type; this
    # override must do the same, or an update whose eltype differs from the cache's
    # (an integer `cache.b = [1, 0, ...]` after the init-time integer-to-float
    # promotion, or Float32 data into a Float64 cache) throws a raw `TypeError`
    # from `setfield!`. The `isa` guard keeps the matching-type hot path (e.g. the
    # warm `cache.A = Awork` refactorization loop, which asserts 0 allocations)
    # from ever reaching `convert`: an already-matching `x` passes through
    # untouched even if constant propagation of `name` fails, so the `===` alias
    # check in the `:A` branch below still sees the caller's object. `:cacheval`
    # is exempt: for the default solver it stores into a slot of the cacheval
    # rather than the field itself, so the field's type is not the right
    # conversion target.
    if name !== :cacheval
        FT = fieldtype(typeof(cache), name)
        x isa FT || (x = convert(FT, x))
    end
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
        update_cacheval!(cache, :A, x)
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
# Integer problems are promoted to float at `init` (the same promotion `\` performs,
# since division does not stay in the integers), so their default tolerance is the
# promoted type's, not the 0 of exact arithmetic. `Rational` stays exact above.
default_tol(::Type{T}) where {T <: Integer} = default_tol(float(T))
default_tol(::Type{Any}) = 0

"""
    __promote_int_arrays(A, b, u0) -> (A, b, u0)

Promote arrays with integer-like element types (`Integer` or
`Complex{<:Integer}`, which includes `Bool` and `BigInt`) the way `\\` does,
and return everything else -- floats, `Rational` (division is closed, solves are
exact), duals, operators -- unchanged, identically (`===`).

Division does not stay in the integers, and LinearSolve factorizes and
back-substitutes into preallocated storage, so without this promotion an integer
`A`, `b`, or `u0` surfaces as an `InexactError` from deep inside `ldiv!` or a
`MethodError` from the QR and Krylov wrappers (issue #206).

The target type is the *joint* promotion of `eltype(A)` and `eltype(b)`, floated
only if that joint type is itself still integer-like. This is what makes mixed
problems match `A \\ b`: an integer `b` against a `Rational` `A` becomes
`Rational` (the solve stays exact), against a `BigFloat` `A` becomes `BigFloat`
(no precision loss), against a `Float32` `A` becomes `Float32`, and only
all-integer problems become `Float64` (`BigInt` becomes `BigFloat`). Conversion
goes through `convert(AbstractArray{T}, x)`, which preserves the container:
structured wrappers (`Symmetric`, `Tridiagonal`, `Diagonal`, `Adjoint`,
`Transpose`), sparse matrices, and static arrays keep their structure, while
`BitArray`s become the `Array`s they must (there is no float bit-array).

Abstractly-typed integer arrays (`Vector{Integer}`, `Union` eltypes) have no
computable `float` eltype, so they promote to the `Float64`/`ComplexF64` default
instead of crashing in `float(::AbstractArray)`.
"""
function __promote_int_arrays(A, b, u0)
    a_int = A isa AbstractArray && _integer_like_eltype(eltype(A))
    b_int = b isa AbstractArray && _integer_like_eltype(eltype(b))
    u_int = u0 isa AbstractArray && _integer_like_eltype(eltype(u0))
    (a_int || b_int || u_int) || return (A, b, u0)
    TA = A isa AbstractArray ? eltype(A) : eltype(b)
    Tb = b isa AbstractArray ? eltype(b) : TA
    T = if isconcretetype(TA) && isconcretetype(Tb)
        Tj = promote_type(TA, Tb)
        _integer_like_eltype(Tj) ? float(Tj) : Tj
    else
        (TA <: Complex || Tb <: Complex) ? ComplexF64 : Float64
    end
    return (
        a_int ? convert(AbstractArray{T}, A) : A,
        b_int ? convert(AbstractArray{T}, b) : b,
        u_int ? convert(AbstractArray{T}, u0) : u0,
    )
end

_integer_like_eltype(::Type{<:Integer}) = true
_integer_like_eltype(::Type{Complex{T}}) where {T} = _integer_like_eltype(T)
_integer_like_eltype(::Type) = false

"""
    __wants_int_promotion(alg)

Whether integer-like `A`/`b`/`u0` should be promoted to float for `alg`; see
[`__promote_int_arrays`](@ref). `false` for the `AbstractSolveFunction` family
(`LinearSolveFunction`, `DirectLdiv!`): those wrap user-supplied solve semantics
-- a `LinearSolveFunction` may implement exact integer or GF(2)/`Bool`
arithmetic, and `DirectLdiv!` calls `ldiv!` on exactly what it was given -- so
LinearSolve must hand them the user's arrays untouched.
"""
__wants_int_promotion(::Union{SciMLLinearSolveAlgorithm, Nothing}) = true
__wants_int_promotion(::AbstractSolveFunction) = false

"""
    __promote_int_problem(prob, alg) -> LinearProblem

Apply [`__promote_int_arrays`](@ref) to a `LinearProblem` before the default
algorithm is chosen. Promotion inside `__init` alone is not enough for
`solve(prob)`: `defaultalg` would select on the unpromoted types while every
cacheval is built from the promoted ones, and the two can disagree -- a
`BitMatrix` or integer `Adjoint`/`Transpose` is not a `DenseMatrix`, so
`defaultalg` picks a Krylov method whose workspace slot is then typed for the
dense float matrix the cache actually holds. Returns `prob` itself (`===`) when
nothing needs promoting.
"""
function __promote_int_problem(prob::LinearProblem, alg)
    __wants_int_promotion(alg) || return prob
    A, b, u0 = __promote_int_arrays(prob.A, prob.b, prob.u0)
    (A === prob.A && b === prob.b && u0 === prob.u0) && return prob
    return SciMLBase.remake(prob; A, b, u0)
end

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

"""
    _check_square_A_support(alg, A)

Throw an informative `ArgumentError` at `init` time when a non-square `A` is
used with an algorithm that requires a square one (see [`needs_square_A`](@ref)).

Without this, a non-square `A` reaches the factorization and fails there, in a
different way for each algorithm and with no indication of what to use instead:
`DimensionMismatch: matrix is not square` from LU and Cholesky, `ArgumentError:
Bunch-Kaufman decomposition is only valid for...`, or `FieldError: type Array has
no field diag` from `DiagonalFactorization`. Least-squares and minimum-norm
systems are solved by the default algorithm and by the algorithms named in the
message.
"""
_check_square_A_support(alg, A) = nothing
function _check_square_A_support(alg::SciMLLinearSolveAlgorithm, A)
    (needs_square_A(alg) && !issquare(A)) && throw(
        ArgumentError(
            "$(nameof(typeof(alg))) requires a square `A`, got $(size(A, 1))x$(size(A, 2)). " *
                "A non-square system is solved in the least-squares (tall `A`) or " *
                "minimum-norm (wide `A`) sense by the default algorithm, i.e. " *
                "`solve(prob)`, or by `QRFactorization(ColumnNorm())`, " *
                "`SVDFactorization()`, and the least-squares Krylov methods " *
                "`KrylovJL_LSMR()` (tall) / `KrylovJL_CRAIGMR()` (wide)."
        )
    )
    return nothing
end

# A right-hand side that implements the SciMLStructures interface but is not an
# array cannot go through `__init` at all: it has no `eltype`, `length` or `similar`,
# so the `abstol`/`reltol`/`maxiters` defaults below have nothing to work from, and it
# fails at `real(eltype(prob.b))` before any algorithm is reached.
#
# `canonicalize` gives a flat buffer of the tunable values that every algorithm can
# handle, plus the `repack` that rebuilds the container. The buffer is built once
# here, and the solve writes back with `replace!` so repeated solves do not allocate
# a new container each time. See SciML/LinearSolve.jl#1208.
_is_structure_rhs(b) = !(b isa AbstractArray) && SciMLStructures.isscimlstructure(b)

"""
    StructureLinearCache(cache, b, repack)

Wraps the flat `LinearCache` a structured right-hand side is actually solved on,
keeping the caller's container and the `repack` that rebuilds it. See
[`_is_structure_rhs`](@ref).
"""
mutable struct StructureLinearCache{C, B, U, R}
    cache::C
    b::B
    # The container the answer is written into, built once so a repeated solve can
    # `replace!` into it. Kept separate from `b`, which must not be overwritten.
    u::U
    repack::R
end

function SciMLBase.init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    if _is_structure_rhs(prob.b)
        buffer, repack, _ = SciMLStructures.canonicalize(
            SciMLStructures.Tunable(), prob.b
        )
        flat = LinearProblem(prob.A, buffer; u0 = zero(buffer), p = prob.p)
        return StructureLinearCache(
            __init(flat, alg, args...; kwargs...), prob.b,
            repack(zero(buffer)), repack
        )
    end
    return __init(prob, alg, args...; kwargs...)
end

# `b` is re-canonicalized when the caller assigns a new container, so an updated
# right-hand side reaches the flat cache the solve runs on.
function Base.setproperty!(cache::StructureLinearCache, sym::Symbol, v)
    if sym === :b
        buffer, repack, _ = SciMLStructures.canonicalize(SciMLStructures.Tunable(), v)
        cache.cache.b = buffer
        setfield!(cache, :repack, repack)
        return setfield!(cache, :b, v)
    elseif sym in (:A, :Pl, :Pr, :abstol, :reltol, :maxiters)
        return setproperty!(cache.cache, sym, v)
    end
    return setfield!(cache, sym, v)
end

function Base.getproperty(cache::StructureLinearCache, sym::Symbol)
    sym in (:cache, :b, :u, :repack) && return getfield(cache, sym)
    return getproperty(getfield(cache, :cache), sym)
end

function SciMLBase.solve!(cache::StructureLinearCache, args...; kwargs...)
    sol = SciMLBase.solve!(getfield(cache, :cache), args...; kwargs...)
    # `replace!` into the solution container where the type allows it, so a repeated
    # solve does not allocate a new one. `b` is never written to.
    u = getfield(cache, :u)
    if SciMLStructures.ismutablescimlstructure(u)
        SciMLStructures.replace!(SciMLStructures.Tunable(), u, sol.u)
    else
        u = getfield(cache, :repack)(sol.u)
        setfield!(cache, :u, u)
    end
    # `build_linear_solution` derives the solution's `N` from `size(u)`, which a
    # container that is not an array does not answer, so the solution is built here
    # with the element type of the canonical buffer and `N = 1`: the tunable portion
    # is a flat vector by the interface's own contract.
    return SciMLBase.LinearSolution{
        eltype(sol.u), 1, typeof(u), typeof(sol.resid), typeof(sol.alg),
        typeof(getfield(cache, :cache)), typeof(sol.stats),
    }(
        u, sol.resid, sol.alg, sol.retcode, sol.iters,
        getfield(cache, :cache), sol.stats
    )
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

    # Integer-eltype problems are solved as their float (or joint-promoted)
    # counterparts, matching `\`; see `__promote_int_arrays`. This happens before
    # the aliasing/copy blocks and before any cacheval is built, so every
    # downstream type sees the promoted problem. `solve(prob)`/`init(prob)` also
    # promote before `defaultalg` (see `__promote_int_problem`), making this a
    # no-op there; doing it here as well covers direct `init(prob, alg)` calls.
    if __wants_int_promotion(alg)
        A, b, u0 = __promote_int_arrays(A, b, u0)
    end
    # The promoted counterpart of `prob.A` (identity when nothing promotes): the
    # DefaultLinearSolver cacheval below types its `A_backup` from this rather
    # than from the working `A`, and it must be the promoted type or the backup
    # copy of the float working matrix throws `InexactError` into integer storage.
    A_original = A

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
    _check_square_A_support(alg, A)

    u0_ = u0 !== nothing ? u0 : __init_u0_from_Ab(A, b)

    # Guard against type mismatch for user-specified reltol/abstol. Use the working
    # `b`, not `prob.b`: an integer `b` has been promoted to float above, and
    # converting a float tolerance to the original integer eltype would throw
    # `InexactError: Int64(1.0e-8)`. Algorithms exempted from promotion
    # (`AbstractSolveFunction`) keep their integer `b`, so an integer eltype is
    # still floated here for the tolerance itself.
    Ttol = let T = real(eltype(b))
        _integer_like_eltype(T) ? float(T) : T
    end
    reltol = Ttol(SciMLBase.value(reltol))
    abstol = Ttol(SciMLBase.value(abstol))

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
    # For DefaultLinearSolver, pass the uncopied original `A` so the A_backup field
    # gets the correct type at construction time (it may be e.g. a WOperator while
    # the converted A used for sub-caches is a different concrete type). This is
    # `A_original`, not `prob.A`: for an integer problem the backup must be typed
    # on the promoted float matrix, or the safety-fallback's `copyto!` of the float
    # working matrix into it throws `InexactError` (first solve for wrappers like
    # `Symmetric{Int}` whose promoted copy holds non-integral values, and any cache
    # reuse that assigns a fractional `A`).
    cacheval = if alg isa DefaultLinearSolver
        init_cacheval(
            alg, A, b, u0_, Pl, Pr, maxiters, abstol, reltol, init_cache_verb,
            assumptions, A_original
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
    # A structured right-hand side has no size or eltype for `defaultalg` to read,
    # so the algorithm is chosen from the canonical buffer it will actually be
    # solved on. `init` canonicalizes again and owns the repack; see
    # `_is_structure_rhs`.
    if _is_structure_rhs(prob.b)
        buffer, = SciMLStructures.canonicalize(SciMLStructures.Tunable(), prob.b)
        return solve(prob, defaultalg(prob.A, buffer, assump), args...; kwargs...)
    end
    # Promote integer-eltype problems before choosing the algorithm, so the choice
    # and the cache agree on the types; see `__promote_int_problem`.
    prob = __promote_int_problem(prob, nothing)
    return solve(prob, defaultalg(prob.A, prob.b, assump), args...; kwargs...)
end

function SciMLBase.solve(
        prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...; kwargs...
    )
    return solve!(init(prob, alg, args...; kwargs...))
end

"""
    solve!(cache::LinearCache, args...; adjoint = false, kwargs...)

`adjoint = true` solves `adjoint(A) x = b` instead of `A x = b`, reusing the
factorization the cache already holds rather than factorizing the adjoint afresh:

```julia
cache = init(LinearProblem(A, b))
x = solve!(cache).u
cache.b = c
lambda = solve!(cache; adjoint = true).u   # adjoint(A) lambda = c, same factorization
```

Algorithms that have not opted into reusing their factorization for an adjoint solve
fall back to factorizing `adjoint(A)`, so the keyword is always answerable. The
solution is written into `cache.u` as usual.
"""
@inline function SciMLBase.solve!(
        cache::LinearCache, args...; adjoint::Bool = false, kwargs...
    )
    adjoint && return _solve_adjoint!(cache, args...; kwargs...)
    return solve!(cache, cache.alg, args...; kwargs...)
end

function _solve_adjoint!(cache::LinearCache, args...; kwargs...)
    # The reuse is only possible once a factorization exists, and a fresh `A` has not
    # been factorized yet.
    cache.isfresh && solve!(cache, cache.alg, args...; kwargs...)
    copyto!(cache.u, _adjoint_solve(cache, cache.b))
    return SciMLBase.build_linear_solution(
        cache.alg, cache.u, nothing, cache; retcode = ReturnCode.Success
    )
end

# Special Case for StaticArrays
const StaticLinearProblem = LinearProblem{
    uType, iip, <:SMatrix,
    <:Union{<:SMatrix, <:SVector},
} where {uType, iip}

function SciMLBase.solve(prob::StaticLinearProblem, args...; kwargs...)
    return SciMLBase.solve(prob, nothing, args...; kwargs...)
end

"""
    __static_default_ldiv(A, b)

Solve for the static-array default algorithm. For square `A`, singular input is
rescued with an SVD least-squares solve (min-norm pseudo-solution), mirroring
the dense default's LU -> pivoted-QR `safetyfallback`. SVD is used instead of
QR because `qr(::SMatrix) \\ b` least-squares is not defined in StaticArrays
(see `defaultalg(::SMatrix)`).
"""
function __static_default_ldiv(A::SMatrix{N, N}, b) where {N}
    # StaticArrays' square `\` uses direct inverse formulas for N <= 3 (never
    # throws; singular input silently yields Inf/NaN) and `lu(A) \ b` with
    # `check = true` (throws SingularException) for larger sizes. Keep `A \ b`
    # for N <= 3 so nonsingular results stay bit-identical to `\`; for larger
    # sizes `lu(A, check = false) \ b` is the same computation as `A \ b`
    # without the throw.
    if N <= 3
        # Fully inlined: StaticArrays' small-size formulas have
        # inlining-context-dependent FMA contraction, so results may differ
        # from a bare `A \ b` in the last bit.
        u = A \ b
        # Only rescue on factorization failure: a nonsingular `A` with
        # non-finite `u` (e.g. non-finite `b`) returns `u` as-is, matching the
        # dense LU behavior.
        if all(isfinite, u) || LinearAlgebra.issuccess(lu(A, check = false))
            return u
        end
    else
        F = lu(A, check = false)
        if LinearAlgebra.issuccess(F)
            return F \ b
        end
    end
    return svd(A) \ b
end
__static_default_ldiv(A, b) = A \ b

function __static_gesv_ldiv(A::SMatrix{N, N}, b) where {N}
    if N <= 3
        u = A \ b
        ok = all(isfinite, u) || LinearAlgebra.issuccess(lu(A, check = false))
        return u, ok
    else
        F = lu(A, check = false)
        ok = LinearAlgebra.issuccess(F)
        # StaticArrays' triangular solve throws on a zero pivot, so the failed
        # branch must not touch `F \ b`. The placeholder mirrors `F \ b`'s
        # promoted eltype to keep the return type-stable; the caller replaces
        # `u` on failure.
        u = ok ? F \ b : zero(b) / oneunit(eltype(A))
        return u, ok
    end
end
function __static_gesv_ldiv(A, b)
    throw(ArgumentError("GESVFactorization requires a square matrix, got size $(size(A))"))
end

function SciMLBase.solve(
        prob::StaticLinearProblem,
        alg::Nothing, args...; kwargs...
    )
    u = __static_default_ldiv(prob.A, prob.b)
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
        F = lu(prob.A, check = false)
        if !LinearAlgebra.issuccess(F)
            # Match dense `LUFactorization` on singular input: report Failure
            # with the (zero-)initialized `u` instead of throwing.
            u = prob.u0 !== nothing ? prob.u0 : __init_u0_from_Ab(prob.A, prob.b)
            return SciMLBase.build_linear_solution(
                alg, u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        u = F \ prob.b
    elseif alg isa GESVFactorization
        # The static analog of the dense `gesv` driver: a one-shot
        # factorize-and-solve whose singular failures are reported through the
        # return code — no SVD rescue (that is the default algorithm's
        # behavior, which gesv semantics do not include). Uses the direct
        # small-size formulas, so this is the fastest explicit static
        # algorithm at small N.
        u, gesv_ok = __static_gesv_ldiv(prob.A, prob.b)
        if !gesv_ok
            u = prob.u0 !== nothing ? prob.u0 : __init_u0_from_Ab(prob.A, prob.b)
            return SciMLBase.build_linear_solution(
                alg, u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
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

"""
    LinearSolve.update_tolerances!(cache; abstol = nothing, reltol = nothing)

Change the convergence tolerances of an existing `LinearCache` in place. The
`abstol`/`reltol` fields are updated and then
[`update_tolerances_internal!`](@ref) gives the algorithm a chance to propagate
the new values into `cache.cacheval`.

Not every algorithm has tolerances to update: factorizations, and algorithms
that do not define `update_tolerances_internal!`, throw instead of silently
ignoring the request.
"""
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
