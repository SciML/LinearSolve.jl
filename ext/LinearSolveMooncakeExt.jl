module LinearSolveMooncakeExt

using Mooncake
using Mooncake: @from_chainrules, MinimalCtx, ReverseMode, NoRData, increment!!, @is_primitive, primal, zero_fcodual, CoDual, rdata, fdata
using LinearSolve: LinearSolve, SciMLLinearSolveAlgorithm, init, solve!, LinearProblem,
    LinearCache, AbstractKrylovSubspaceMethod, DefaultLinearSolver, LinearSolveAdjoint,
    defaultalg_adjoint_eval, solve, LUFactorization
using LinearSolve.LinearAlgebra
using SciMLBase

@from_chainrules MinimalCtx Tuple{typeof(SciMLBase.solve), LinearProblem, Nothing} true ReverseMode
@from_chainrules MinimalCtx Tuple{
    typeof(SciMLBase.solve), LinearProblem, SciMLLinearSolveAlgorithm,
} true ReverseMode
@from_chainrules MinimalCtx Tuple{
    Type{<:LinearProblem}, AbstractMatrix, AbstractVector, SciMLBase.NullParameters,
} true ReverseMode

function Mooncake.increment_and_get_rdata!(f, r::NoRData, t::LinearProblem)
    f.data.A .+= t.A
    f.data.b .+= t.b

    return NoRData()
end

function Mooncake.increment_and_get_rdata!(f, r::NoRData, t::LinearCache)
    f.fields.A .+= t.A
    f.fields.b .+= t.b
    f.fields.u .+= t.u

    return NoRData()
end

# rrules for LinearCache
@from_chainrules MinimalCtx Tuple{typeof(init), LinearProblem, SciMLLinearSolveAlgorithm} true ReverseMode
@from_chainrules MinimalCtx Tuple{typeof(init), LinearProblem, Nothing} true ReverseMode

# rrules for solve!
# NOTE - Avoid Mooncake.prepare_gradient_cache, only use Mooncake.prepare_pullback_cache (and therefore Mooncake.value_and_pullback!!)
# calling Mooncake.prepare_gradient_cache for functions with solve! will activate unsupported Adjoint case exception for below rrules
# This is because in Mooncake.prepare_gradient_cache we reset stacks + state by passing in zero gradient in the reverse pass once.
# However, if one has a valid cache then they can directly use Mooncake.value_and_gradient!!.

@is_primitive MinimalCtx Tuple{typeof(SciMLBase.solve!), LinearCache, SciMLLinearSolveAlgorithm, Vararg}
@is_primitive MinimalCtx Tuple{typeof(SciMLBase.solve!), LinearCache, Nothing, Vararg}

function Mooncake.rrule!!(sig::CoDual{typeof(SciMLBase.solve!)}, _cache::CoDual{<:LinearSolve.LinearCache}, _alg::CoDual{Nothing}, args::Vararg{Any, N}; kwargs...) where {N}
    cache = primal(_cache)
    assump = OperatorAssumptions()
    _alg.x = defaultalg(cache.A, cache.b, assump)
    return Mooncake.rrule!!(sig, _cache, _alg, args...; kwargs...)
end

function Mooncake.rrule!!(
        ::CoDual{typeof(SciMLBase.solve!)}, _cache::CoDual{<:LinearSolve.LinearCache}, _alg::CoDual{<:SciMLLinearSolveAlgorithm}, args::Vararg{Any, N}; alias_A = zero_fcodual(
            LinearSolve.default_alias_A(
                _alg.x, _cache.x.A, _cache.x.b
            )
        ), kwargs...
    ) where {N}

    cache = primal(_cache)
    alg = primal(_alg)
    _args = map(primal, args)

    (; A, b, sensealg) = cache
    A_orig = copy(A)
    b_orig = copy(b)

    @assert sensealg isa LinearSolveAdjoint "Currently only `LinearSolveAdjoint` is supported for adjoint sensitivity analysis."

    A_ = nothing
    if sensealg.linsolve === missing
        can_reuse_factorization = LinearSolve._can_reuse_cache_factorization(
            alg, cache.cacheval
        )
        if !(
                can_reuse_factorization || alg isa LinearSolve.AbstractKrylovSubspaceMethod ||
                    alg isa LinearSolve.DefaultLinearSolver
            )
            A_ = alg isa LinearSolve.AbstractFactorization ? deepcopy(A) :
                alias_A ? deepcopy(A) : A
        end
    else
        A_ = deepcopy(A)
    end

    sol = zero_fcodual(solve!(cache))
    cache.A = A_orig
    cache.b = b_orig

    function solve!_adjoint(::NoRData)
        ∂∅ = NoRData()
        alg = cache.alg
        cachenew = init(LinearProblem(cache.A, cache.b), alg, _args...; kwargs...)
        new_sol = solve!(cachenew)
        ∂u = sol.dx.data.u

        if sensealg.linsolve === missing
            cached_adjoint_solution = LinearSolve._adjoint_factorization_solve(
                alg, cache.cacheval, cache.A, ∂u
            )
            λ = if cached_adjoint_solution !== nothing
                cached_adjoint_solution
            elseif alg isa AbstractKrylovSubspaceMethod
                LinearSolve._adjoint_krylov_solve(
                    alg, cache.A, ∂u; cache.abstol, cache.reltol, cache.verbose
                )
            elseif alg isa DefaultLinearSolver
                LinearSolve.defaultalg_adjoint_eval(cache, ∂u)
            else
                invprob = LinearProblem(adjoint(A_), ∂u) # We cached `A`
                solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
            end
        else
            invprob = LinearProblem(adjoint(A_), ∂u) # We cached `A`
            λ = solve(
                invprob, sensealg.linsolve; cache.abstol, cache.reltol, cache.verbose
            ).u
        end

        tu = adjoint(new_sol.u)
        ∂A = .-(λ .* tu)
        ∂b = λ

        if (iszero(∂b) || iszero(∂A)) && !iszero(tu)
            error("Adjoint case currently not handled. Instead of using `solve!(cache); s1 = copy(cache.u) ...`, use `sol = solve!(cache); s1 = copy(sol.u)`.")
        end

        fdata(_cache.dx).fields.A .+= ∂A
        fdata(_cache.dx).fields.b .+= ∂b
        fdata(_cache.dx).fields.u .+= ∂u

        # rdata for cache is a struct with NoRdata field values
        return (∂∅, rdata(_cache.dx), ∂∅, ntuple(_ -> ∂∅, length(args))...)
    end

    return sol, solve!_adjoint
end

end
