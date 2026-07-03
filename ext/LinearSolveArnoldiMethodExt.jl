module LinearSolveArnoldiMethodExt

using LinearAlgebra
using LinearSolve
import ArnoldiMethod: partialschur, partialeigen
using SciMLBase: SciMLBase, ReturnCode

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.ArnoldiMethodJL,
        args...; kwargs...)
    prob.B === nothing ||
        error("ArnoldiMethod backend currently supports standard eigenvalue problems only.")
    nev = LinearSolve.default_nev(prob)
    which = prob.sigma === nothing ? prob.which : :LM
    A = prob.sigma === nothing ? prob.A : _shift_invert_operator(prob.A, prob.sigma)
    kw = (; nev, which, prob.kwargs..., alg.kwargs..., kwargs...)
    decomp, history = partialschur(A, alg.args...; kw...)
    values, vectors = partialeigen(decomp)
    if prob.sigma !== nothing
        values = prob.sigma .+ inv.(values)
    end
    values, vectors = LinearSolve._select_eigenpairs(values, vectors, nev, prob.which, prob.sigma)
    retcode = history.converged ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    return LinearSolve.build_eigenvalue_solution(
        prob, alg, values, vectors; retcode, stats = history)
end

function _shift_invert_operator(A, sigma)
    F = factorize(A - sigma * I)
    T = promote_type(eltype(A), typeof(sigma))
    return ShiftInvertMap{typeof(F), T}(F, size(A, 1))
end

struct ShiftInvertMap{F, T}
    F::F
    n::Int
end

Base.size(A::ShiftInvertMap) = (A.n, A.n)
Base.size(A::ShiftInvertMap, dim::Integer) = dim <= 2 ? A.n : 1
Base.eltype(::Type{<:ShiftInvertMap{F, T}}) where {F, T} = T

function LinearAlgebra.mul!(y, A::ShiftInvertMap, x)
    copyto!(y, A.F \ x)
    return y
end

end
