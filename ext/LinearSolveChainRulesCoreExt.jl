module LinearSolveChainRulesCoreExt

using LinearSolve: LinearSolve, SciMLLinearSolveAlgorithm, AbstractFactorization,
    AbstractKrylovSubspaceMethod, DefaultLinearSolver, OperatorAssumptions,
    NormalBunchKaufmanFactorization, NormalCholeskyFactorization,
    defaultalg, default_alias_A, LinearSolveAdjoint
using SciMLBase: SciMLBase, LinearProblem, init, solve, solve!
using SciMLOperators: issquare
using ChainRulesCore: ChainRulesCore, NoTangent
using LinearAlgebra: adjoint

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

    A_ = nothing
    if sensealg.linsolve === missing
        can_reuse_factorization = LinearSolve._can_reuse_cache_factorization(
            alg, cache.cacheval
        )
        if !(
                can_reuse_factorization || alg isa AbstractKrylovSubspaceMethod ||
                    alg isa DefaultLinearSolver
            )
            A_ = if alg isa AbstractFactorization
                deepcopy(A)
            else
                alias_A ? deepcopy(A) : A
            end
        end
    else
        A_ = deepcopy(A)
    end

    # The correction for a non-square `A` reads `A` and `b`, which the solve may overwrite.
    nonsquare = !LinearSolve.issquare(A) && A isa AbstractMatrix
    A_ns = if !nonsquare || alg isa Union{
            AbstractKrylovSubspaceMethod, NormalCholeskyFactorization,
            NormalBunchKaufmanFactorization,
        }
        A  # never written to
    elseif A_ !== nothing
        A_
    elseif A isa Matrix && A !== prob.A
        prob.A  # `init` made `A` a private copy
    else
        deepcopy(A)
    end
    b_ = nonsquare ? copy(cache.b) : nothing

    sol = solve!(cache)

    function ∇linear_solve(∂sol)
        ∂∅ = NoTangent()

        ∂u = hasproperty(∂sol, :u) ? ∂sol.u : ∂sol
        if sensealg.linsolve === missing
            # Same route as `solve!(cache; adjoint = true)`. `A_` is the copy preserved
            # above when the factorization may have overwritten `cache.A`.
            λ = LinearSolve._adjoint_solve(cache, ∂u, A_ === nothing ? cache.A : A_)
        else
            adj_Pl, adj_Pr = LinearSolve._adjoint_precs(
                sensealg.linsolve, sensealg, cache.Pl, cache.Pr
            )
            invprob = LinearProblem(adjoint(A_), ∂u) # We cached `A`
            λ = solve(
                invprob, sensealg.linsolve; cache.abstol, cache.reltol, cache.verbose,
                Pl = adj_Pl, Pr = adj_Pr
            ).u
        end

        tu = adjoint(sol.u)
        ∂A = .-(λ .* tu)
        LinearSolve._add_nonsquare_pullback!(∂A, cache, A_ns, b_, sol.u, λ, ∂u)
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
