module LinearSolveArpackExt

using LinearSolve
using Arpack
using SciMLBase: SciMLBase, ReturnCode

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.ArpackJL,
        args...; kwargs...
    )
    nev = LinearSolve.default_num_eigenpairs(prob)
    base = (; nev, which = _arpack_which(prob.eigentarget))
    if prob.shift !== nothing
        base = (; base..., sigma = prob.shift)
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

# Arpack.eigs requires a raw ARPACK-style Symbol for `which`; this mapping is
# purely a private adapter to that third-party API, not a general LinearSolve
# concept.
function _arpack_which(w::LinearSolve.EigenvalueTarget.T)
    T = LinearSolve.EigenvalueTarget
    return w == T.LargestMagnitude ? :LM :
        w == T.SmallestMagnitude ? :SM :
        w == T.LargestRealPart ? :LR :
        w == T.SmallestRealPart ? :SR :
        w == T.LargestImaginaryPart ? :LI :
        :SI
end

end
