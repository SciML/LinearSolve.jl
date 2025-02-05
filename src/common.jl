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
    verbose::Bool
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

init_cacheval(alg::SciMLLinearSolveAlgorithm, args...) = nothing

function SciMLBase.init(prob::LinearProblem, args...; kwargs...)
    SciMLBase.init(prob, nothing, args...; kwargs...)
end

default_tol(::Type{T}) where {T} = √(eps(T))
default_tol(::Type{Complex{T}}) where {T} = √(eps(T))
default_tol(::Type{<:Rational}) = 0
default_tol(::Type{<:Integer}) = 0
default_tol(::Type{Any}) = 0

default_alias_A(::Any, ::Any, ::Any) = false
default_alias_b(::Any, ::Any, ::Any) = false

# Non-destructive algorithms default to true
default_alias_A(::AbstractKrylovSubspaceMethod, ::Any, ::Any) = true
default_alias_b(::AbstractKrylovSubspaceMethod, ::Any, ::Any) = true

default_alias_A(::AbstractSparseFactorization, ::Any, ::Any) = true
default_alias_b(::AbstractSparseFactorization, ::Any, ::Any) = true

DEFAULT_PRECS(A, p) = IdentityOperator(size(A)[1]), IdentityOperator(size(A)[2])

function __init_u0_from_Ab(A, b)
    u0 = similar(b, size(A, 2))
    fill!(u0, false)
    return u0
end
__init_u0_from_Ab(::SMatrix{S1, S2}, b) where {S1, S2} = zeros(SVector{S2, eltype(b)})

function SciMLBase.init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
        args...;
        alias = LinearAliasSpecifier(),
        abstol = default_tol(real(eltype(prob.b))),
        reltol = default_tol(real(eltype(prob.b))),
        maxiters::Int = length(prob.b),
        verbose::Bool = false,
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
            please use an ODEAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_A = true))"
            Base.depwarn(message, :init)
            Base.depwarn(message, :solve)
            aliases = LinearAliasSpecifier(alias_A = values(kwargs).alias_A)
        end

        if haskey(kwargs, :alias_b)
            message = "`alias_b` keyword argument is deprecated, to set `alias_b`,
            please use an ODEAliasSpecifier, e.g. `solve(prob, alias = LinearAliasSpecifier(alias_b = true))"
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
        SparseMatrixCSC(size(b)..., getcolptr(b), rowvals(b), nonzeros(b))
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
        reinit_cache = false,
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
    if reinit_cache
        return LinearCache{
            typeof(A), typeof(b), typeof(u), typeof(p), typeof(alg), typeof(cacheval),
            typeof(Pl), typeof(Pr), typeof(reltol), typeof(assumptions.issq),
            typeof(sensealg)}(
            A, b, u, p, alg, cacheval, precsisfresh, isfresh, Pl, Pr, abstol, reltol,
            maxiters, verbose, assumptions, sensealg)
    else
        cache.A = A
        cache.b = b
        cache.u = u
        cache.p = p
        cache.Pl = Pl
        cache.Pr = Pr
        cache.isfresh = true
        cache.precsisfresh = precsisfresh
    end
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
    return SciMLBase.build_linear_solution(alg, u, nothing, prob)
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
    return SciMLBase.build_linear_solution(alg, u, nothing, prob)
end
