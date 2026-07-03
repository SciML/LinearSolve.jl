module LinearSolveJacobiDavidsonExt

using LinearSolve
using LinearAlgebra
using JacobiDavidson
using SciMLBase: SciMLBase, ReturnCode

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.JacobiDavidsonJL,
        args...; kwargs...)
    # JacobiDavidson.jl's `jdqz` (generalized solver) is broken upstream: it
    # references an undefined `verbose` variable that is absent from its
    # signature. Until that is fixed, only the standard solver `jdqr` is wired
    # up here; point users at the backends that do support generalized problems.
    prob.B === nothing ||
        error("The JacobiDavidson backend currently supports standard eigenvalue problems only. Use `ArpackJL()` or `KrylovKitEigen()` for generalized problems.")

    n = size(prob.A, 2)
    nev = LinearSolve.default_nev(prob)
    target = _jd_target(prob)
    # Search-subspace bounds, capped at the problem size. Users may override
    # `subspace_dimensions` (and any other jdqr keyword) via the algorithm.
    hi = min(max(2 * nev + 10, 20), n)
    lo = min(max(nev + 2, 8), hi)
    defaults = (; pairs = nev, target = target, subspace_dimensions = lo:hi)
    kw = (; defaults..., prob.kwargs..., alg.kwargs..., kwargs...)

    out = JacobiDavidson.jdqr(prob.A, alg.args...; kw...)
    values, vectors = _jd_standard_pairs(prob.A, out[1])

    values, vectors = LinearSolve._select_eigenpairs(
        values, vectors, nev, prob.which, prob.sigma)
    retcode = length(values) >= nev ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    return LinearSolve.build_eigenvalue_solution(
        prob, alg, values, vectors; retcode, stats = out[end])
end

# Map the problem's spectral selector onto a JacobiDavidson `Target`. A supplied
# `sigma` is the natural interior target (`Near`), which is Jacobi-Davidson's
# strength; otherwise the `which` symbol selects an extremal target.
function _jd_target(prob)
    prob.sigma !== nothing && return JacobiDavidson.Near(ComplexF64(prob.sigma))
    w = prob.which
    w === :LM ? JacobiDavidson.LargestMagnitude(0.0 + 0.0im) :
    w === :SM ? JacobiDavidson.SmallestMagnitude(0.0 + 0.0im) :
    w === :LR ? JacobiDavidson.LargestRealPart(0.0 + 0.0im) :
    w === :SR ? JacobiDavidson.SmallestRealPart(0.0 + 0.0im) :
    w === :LI ? JacobiDavidson.LargestImaginaryPart(0.0 + 0.0im) :
    w === :SI ? JacobiDavidson.SmallestImaginaryPart(0.0 + 0.0im) :
    throw(ArgumentError("unsupported `which = $(w)` for JacobiDavidson; expected one of :LM, :SM, :LR, :SR, :LI, :SI, or pass `sigma`"))
end

# jdqr yields a partial Schur decomposition `A*Q = Q*R`. Eigenpairs are recovered
# from the small projected `R = Q'AQ`: if `R*y = λ*y` then `A*(Q*y) = λ*(Q*y)`.
function _jd_standard_pairs(A, pschur)
    T = complex(float(eltype(A)))
    k = length(pschur.values)
    k == 0 && return (T[], Matrix{T}(undef, size(A, 1), 0))
    Q = pschur.Q[:, 1:k]
    F = eigen(Q' * (A * Q))
    return (F.values, Q * F.vectors)
end

end
