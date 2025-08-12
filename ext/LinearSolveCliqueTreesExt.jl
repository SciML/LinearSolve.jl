module LinearSolveCliqueTreesExt

using CliqueTrees: EliminationAlgorithm, SupernodeType, DEFAULT_ELIMINATION_ALGORITHM,
    DEFAULT_SUPERNODE_TYPE, symbolic, cholinit, lininit, cholesky!, linsolve!
using LinearSolve

function LinearSolve.CliqueTreesFactorization(;
        alg::A=DEFAULT_ELIMINATION_ALGORITHM,
        snd::S=DEFAULT_SUPERNODE_TYPE, 
        reuse_symbolic::Bool=true,
    ) where {A <: EliminationAlgorithm, S <: SupernodeType}
    return CliqueTreesFactorization{A, S}(alg, snd, reuse_symbolic)
end

function LinearSolve.init_cacheval(
    alg::CliqueTreesFactorization, A::AbstractMatrix, b, u, Pl, Pr, maxiters::Int, abstol,
    reltol, verbose::Bool, assumptions::OperatorAssumptions)
    symbfact = symbolic(A; alg=alg.alg, snd=alg.snd)
    cholfact, cholwork = cholinit(A, symbfact)
    linwork = lininit(1, cholfact)
    return (cholfact, cholwork, linwork)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CliqueTreesFactorization; kwargs...)
    A = cache.A
    u = cache.u
    b = cache.b

    if cache.isfresh
        if isnothing(cache.cacheval) || !alg.reuse_symbolic
            symbfact = symbolic(A; alg=alg.alg, snd=alg.snd)
            cholfact, cholwork = cholinit(A, symbfact)
            linwork = lininit(1, cholfact)
            cache.cacheval = (cholfact, cholwork, linwork)
        end

        cholfact, cholwork, linwork = cache.cacheval
        cholesky!(cholfact, cholwork, A)
        cache.isfresh = false
    end

    cholfact, cholwork, linwork = cache.cacheval
    linsolve!(copyto!(u, b), linwork, cholfact, Val(false))
    return SciMLBase.build_linear_solution(alg, u, nothing, cache) 
end

LinearSolve.PrecompileTools.@compile_workload begin
    A = [
        3 1 0 0 0 0 0 0
        1 3 1 0 0 2 0 0
        0 1 3 1 0 1 2 1
        0 0 1 3 0 0 0 0
        0 0 0 0 3 1 1 0
        0 2 1 0 1 3 0 0
        0 0 2 0 1 0 3 1
        0 0 1 0 0 0 1 3
    ];

    b = rand(8)
    prob = LinearProblem(A, b)
    sol = solve(prob) # in case cliquetrees is used as default
    sol = solve(prob, CliqueTreesFactorization())
end

end
