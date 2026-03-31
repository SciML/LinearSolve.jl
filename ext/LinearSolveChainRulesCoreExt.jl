module LinearSolveChainRulesCoreExt

using LinearSolve
using LinearSolve: SciMLLinearSolveAlgorithm, AbstractFactorization,
    AbstractKrylovSubspaceMethod, DefaultLinearSolver, OperatorAssumptions,
    defaultalg, default_alias_A, defaultalg_adjoint_eval, LinearSolveAdjoint
using SciMLBase: SciMLBase, LinearProblem, init, solve, solve!
using SciMLOperators: issquare
using ChainRulesCore: ChainRulesCore, NoTangent
using LinearAlgebra: Factorization, adjoint

const CRC = ChainRulesCore

function CRC.rrule(
        T::typeof(SciMLBase.solve), prob::LinearProblem, alg::Nothing, args...; kwargs...
    )
    assump = OperatorAssumptions(issquare(prob.A))
    alg = defaultalg(prob.A, prob.b, assump)
    return CRC.rrule(T, prob, alg, args...; kwargs...)
end

function CRC.rrule(
        ::typeof(SciMLBase.solve), prob::LinearProblem,
        alg::SciMLLinearSolveAlgorithm, args...; alias_A = default_alias_A(
            alg, prob.A, prob.b
        ), kwargs...
    )
    # sol = solve(prob, alg, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    (; A, sensealg) = cache

    @assert sensealg isa LinearSolveAdjoint "Currently only `LinearSolveAdjoint` is supported for adjoint sensitivity analysis."

    # Decide if we need to cache `A` and `b` for the reverse pass
    if sensealg.linsolve === missing
        # We can reuse the factorization so no copy is needed
        # Krylov Methods don't modify `A`, so it's safe to just reuse it
        # No Copy is needed even for the default case
        if !(
                alg isa AbstractFactorization || alg isa AbstractKrylovSubspaceMethod ||
                    alg isa DefaultLinearSolver
            )
            A_ = alias_A ? deepcopy(A) : A
        end
    else
        A_ = deepcopy(A)
    end

    sol = solve!(cache)

    function ∇linear_solve(∂sol)
        ∂∅ = NoTangent()

        ∂u = hasproperty(∂sol, :u) ? ∂sol.u : ∂sol
        if sensealg.linsolve === missing
            λ = if cache.cacheval isa Factorization
                cache.cacheval' \ ∂u
            elseif cache.cacheval isa Tuple && cache.cacheval[1] isa Factorization
                first(cache.cacheval)' \ ∂u
            elseif alg isa AbstractKrylovSubspaceMethod
                invprob = LinearProblem(adjoint(cache.A), ∂u)
                solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
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

        tu = adjoint(sol.u)
        ∂A = .-(λ .* tu)
        ∂b = λ
        ∂prob = LinearProblem(∂A, ∂b, ∂∅)

        return (∂∅, ∂prob, ∂∅, ntuple(_ -> ∂∅, length(args))...)
    end

    return sol, ∇linear_solve
end

function CRC.rrule(::Type{<:LinearProblem}, A, b, p; kwargs...)
    prob = LinearProblem(A, b, p)
    ∇prob(∂prob) = (NoTangent(), ∂prob.A, ∂prob.b, ∂prob.p)
    return prob, ∇prob
end

function CRC.rrule(T::typeof(LinearSolve.init), prob::LinearSolve.LinearProblem, alg::Nothing, args...; kwargs...)
    assump = OperatorAssumptions(issquare(prob.A))
    alg = defaultalg(prob.A, prob.b, assump)
    return CRC.rrule(T, prob, alg, args...; kwargs...)
end

function CRC.rrule(::typeof(LinearSolve.init), prob::LinearSolve.LinearProblem, alg::Union{LinearSolve.SciMLLinearSolveAlgorithm, Nothing}, args...; kwargs...)
    init_res = LinearSolve.init(prob, alg)
    function init_adjoint(∂init)
        ∂prob = LinearProblem(∂init.A, ∂init.b, NoTangent())
        return NoTangent(), ∂prob, NoTangent(), ntuple((_ -> NoTangent(), length(args))...)
    end

    return init_res, init_adjoint
end

end
