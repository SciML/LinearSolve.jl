module LinearSolveAlgebraicMultigridExt

using LinearSolve, AlgebraicMultigrid, LinearAlgebra
using LinearSolve: LinearCache, LinearVerbosity, OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode

function LinearSolve.init_cacheval(
    alg::AlgebraicMultigridJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
    verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
)
    @assert size(A, 1) == size(A, 2) "AlgebraicMultigrid.jl requires a square matrix"
    
    method = if isempty(alg.args)
        AlgebraicMultigrid.ruge_stuben
    else
        alg.args[1]
    end
    
    return method(A; alg.kwargs...)
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
    
    ml = cache.cacheval

    # Use the tolerances from the cache
    tol = cache.reltol
    maxiter = cache.maxiters
    
    # AlgebraicMultigrid.jl doesn't export a public API for solving with a precomputed 
    # MultiLevel object without rebuilding it or using the CommonSolve interface which 
    # implies creating a new AMGSolver wrapper.
    # However, `_solve!` is the internal function that does exactly what we want:
    # in-place solve using the existing hierarchy.
    AlgebraicMultigrid._solve!(cache.u, ml, cache.b; 
        maxiter=maxiter, 
        reltol=tol
    )
    
    # Basic return code handling (AMG doesn't always return stats in the simple call)
    # We assume success if it finishes.
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, cache; retcode=ReturnCode.Success)
end

end
