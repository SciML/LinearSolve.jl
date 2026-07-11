module LinearSolveArnoldiMethodExt

using LinearAlgebra
using LinearSolve
import ArnoldiMethod: partialschur, partialeigen, LM, LR, SR, LI, SI
using SciMLBase: SciMLBase, ReturnCode

function SciMLBase.solve(
        prob::LinearSolve.EigenvalueProblem,
        alg::LinearSolve.ArnoldiMethodJL,
        args...; kwargs...
    )
    prob.B === nothing ||
        error("ArnoldiMethod backend currently supports standard eigenvalue problems only.")
    nev = LinearSolve.default_num_eigenpairs(prob)
    which = prob.shift === nothing ? _arnoldi_target(prob.eigentarget) : LM()
    A = prob.shift === nothing ? prob.A : _shift_invert_operator(prob.A, prob.shift)
    kw = (; nev, which, prob.kwargs..., alg.kwargs..., kwargs...)
    decomp, history = partialschur(A; kw...)
    values, vectors = partialeigen(decomp)
    if prob.shift !== nothing
        values = prob.shift .+ inv.(values)
    end
    values, vectors = LinearSolve._select_eigenpairs(
        values, vectors, nev, prob.eigentarget, prob.shift
    )
    retcode = history.converged ? ReturnCode.Success : ReturnCode.ConvergenceFailure
    return LinearSolve.build_eigenvalue_solution(
        prob, alg, values, vectors; retcode, stats = history
    )
end

# ArnoldiMethod exposes its own `Target` types (preferred over ARPACK-style
# symbols per its own documentation) for all but smallest-magnitude, which it
# does not support at all.
function _arnoldi_target(w::LinearSolve.EigenvalueTarget.T)
    T = LinearSolve.EigenvalueTarget
    return w == T.LargestMagnitude ? LM() :
        w == T.LargestRealPart ? LR() :
        w == T.SmallestRealPart ? SR() :
        w == T.LargestImaginaryPart ? LI() :
        w == T.SmallestImaginaryPart ? SI() :
        throw(ArgumentError("ArnoldiMethod does not support `eigentarget = EigenvalueTarget.SmallestMagnitude`; use a different backend (e.g. `KrylovKitEigen()` or `ArpackJL()`) or supply `shift` for shift-and-invert."))
end

function _shift_invert_operator(A, shift)
    F = factorize(A - shift * I)
    T = promote_type(eltype(A), typeof(shift))
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
