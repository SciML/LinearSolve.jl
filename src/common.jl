"""
`OperatorCondition`

Specifies the assumption of matrix conditioning for the default linear solver choices. Condition number
is defined as the ratio of eigenvalues. The numerical stability of many linear solver algorithms
can be dependent on the condition number of the matrix. The condition number can be computed as:

```julia
using LinearAlgebra
cond(rand(100,100))
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

mutable struct LinearCache{TA, Tb, Tu, Tp, Talg, Tc, Tl, Tr, Ttol, issq}
    A::TA
    b::Tb
    u::Tu
    p::Tp
    alg::Talg
    cacheval::Tc  # store alg cache here
    isfresh::Bool # false => cacheval is set wrt A, true => update cacheval wrt A
    Pl::Tl        # preconditioners
    Pr::Tr
    abstol::Ttol
    reltol::Ttol
    maxiters::Int
    verbose::Bool
    assumptions::OperatorAssumptions{issq}
end

function Base.setproperty!(cache::LinearCache, name::Symbol, x)
    if name === :A
        setfield!(cache, :isfresh, true)
    elseif name === :cacheval && cache.alg isa DefaultLinearSolver
        @assert cache.cacheval isa DefaultLinearSolverInit
        return setfield!(cache.cacheval, Symbol(cache.alg.alg), x)
    end
    setfield!(cache, name, x)
end

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

function SciMLBase.init(prob::LinearProblem, alg::SciMLLinearSolveAlgorithm,
    args...;
    alias_A = default_alias_A(alg, prob.A, prob.b),
    alias_b = default_alias_b(alg, prob.A, prob.b),
    abstol = default_tol(eltype(prob.A)),
    reltol = default_tol(eltype(prob.A)),
    maxiters::Int = length(prob.b),
    verbose::Bool = false,
    Pl = IdentityOperator(size(prob.A)[1]),
    Pr = IdentityOperator(size(prob.A)[2]),
    assumptions = OperatorAssumptions(issquare(prob.A)),
    kwargs...)
    @unpack A, b, u0, p = prob

    A = if alias_A
        A
    elseif A isa Array || A isa SparseMatrixCSC
        copy(A)
    else
        deepcopy(A)
    end

    b = if b isa SparseArrays.AbstractSparseArray && !(A isa Diagonal)
        Array(b) # the solution to a linear solve will always be dense!
    elseif alias_b
        b
    elseif b isa Array || b isa SparseMatrixCSC
        copy(b)
    else
        deepcopy(b)
    end

    u0 = if u0 !== nothing
        u0
    else
        u0 = similar(b, size(A, 2))
        fill!(u0, false)
    end

    # Guard against type mismatch for user-specified reltol/abstol
    reltol = eltype(prob.A)(reltol)
    abstol = eltype(prob.A)(abstol)

    cacheval = init_cacheval(alg, A, b, u0, Pl, Pr, maxiters, abstol, reltol, verbose,
        assumptions)
    isfresh = true
    Tc = typeof(cacheval)

    cache = LinearCache{
        typeof(A),
        typeof(b),
        typeof(u0),
        typeof(p),
        typeof(alg),
        Tc,
        typeof(Pl),
        typeof(Pr),
        typeof(reltol),
        typeof(assumptions.issq),
    }(A,
        b,
        u0,
        p,
        alg,
        cacheval,
        isfresh,
        Pl,
        Pr,
        abstol,
        reltol,
        maxiters,
        verbose,
        assumptions)
    return cache
end

function SciMLBase.solve(prob::LinearProblem, args...; kwargs...)
    solve!(init(prob, nothing, args...; kwargs...))
end

function SciMLBase.solve(prob::LinearProblem,
    alg::Union{SciMLLinearSolveAlgorithm, Nothing},
    args...; kwargs...)
    solve!(init(prob, alg, args...; kwargs...))
end

function SciMLBase.solve!(cache::LinearCache, args...; kwargs...)
    solve!(cache, cache.alg, args...; kwargs...)
end
