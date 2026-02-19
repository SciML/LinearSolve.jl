module LinearSolveAlgebraicMultigridExt

using LinearSolve, AlgebraicMultigrid, LinearAlgebra
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode

function LinearSolve.init_cacheval(
    alg::AlgebraicMultigridJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
)
    @assert size(A, 1) == size(A, 2) "AlgebraicMultigrid.jl requires a square matrix"

    amg_alg = if isempty(alg.args)
        AlgebraicMultigrid.RugeStubenAMG()
    else
        alg.args[1]
    end

    return SciMLBase.init(amg_alg, A, b; alg.kwargs...)
end

function SciMLBase.solve!(cache::LinearCache, alg::AlgebraicMultigridJL; kwargs...)
    if cache.isfresh
        cache.cacheval = LinearSolve.init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol, cache.verbose,
            cache.assumptions
        )
        cache.isfresh = false
    end

    amg_solver = cache.cacheval

    tol = cache.reltol
    maxiter = cache.maxiters

    # Update b in the solver to reflect any changes since init
    copyto!(amg_solver.b, cache.b)

    x = SciMLBase.solve!(amg_solver; maxiter = maxiter, reltol = tol)
    copyto!(cache.u, x)

    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = ReturnCode.Success)
end

end
