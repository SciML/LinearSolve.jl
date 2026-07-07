module LinearSolveArpackExt

using LinearSolve
using Arpack
using SciMLBase: SciMLBase, ReturnCode

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.ArpackJL,
        args...; kwargs...
    )
    nev = LinearSolve.default_nev(prob)
    base = (; nev, which = LinearSolve._target_symbol(prob.which))
    if prob.sigma !== nothing
        base = (; base..., sigma = prob.sigma)
    end
    kw = (; base..., prob.kwargs..., alg.kwargs..., kwargs...)
    # `Arpack.eigs` takes the generalized-problem matrix `B` positionally, not
    # as a keyword argument.
    values, vectors, nconv, niter, nmult, resid = if prob.B === nothing
        Arpack.eigs(prob.A; kw...)
    else
        Arpack.eigs(prob.A, prob.B; kw...)
    end
    retcode = nconv >= length(values) ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    stats = (; nconv, niter, nmult)
    return LinearSolve.build_eigenvalue_solution(
        prob, alg, values, vectors; retcode, resid, stats
    )
end

end
