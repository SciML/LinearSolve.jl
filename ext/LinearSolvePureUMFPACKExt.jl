module LinearSolvePureUMFPACKExt

using LinearSolve: LinearSolve, PureUMFPACKFactorization, OperatorAssumptions,
    LinearVerbosity
using PureUMFPACK: PureUMFPACK, PureLU, splu
using SparseArrays: SparseArrays, AbstractSparseArray, SparseMatrixCSC,
    nonzeros, rowvals, getcolptr
using SciMLOperators: AbstractSciMLOperator, has_concretization
using SciMLLogging: @SciMLMessage
using SciMLBase: SciMLBase, ReturnCode
using LinearAlgebra: diag

# PureUMFPACK has no preallocatable / in-place-refactorable factorization object
# (no `lu!`-style API): a fresh `splu` recomputes ordering + symbolic + numerics.
# `init_cacheval` returns an empty typed `PureLU` purely to fix the cache field's
# type so `solve!` can store the real factorization; the first solve factorizes.

function _empty_purelu(::Type{Tv}, ::Type{Ti}) where {Tv, Ti}
    Tr = real(Tv)
    empty = SparseMatrixCSC{Tv, Ti}(0, 0, Ti[1], Ti[], Tv[])
    return PureLU{Tv, Ti, Tr}(empty, empty, Ti[], Ti[], Tr[], empty)
end

function LinearSolve.init_cacheval(
        alg::PureUMFPACKFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        alg::PureUMFPACKFactorization, A::AbstractSparseArray{Tv, Ti}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    ) where {Tv, Ti <: Integer}
    return _empty_purelu(Tv, Ti)
end

function LinearSolve.init_cacheval(
        alg::PureUMFPACKFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if has_concretization(A)
        return LinearSolve.init_cacheval(
            alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
            maxiters, abstol, reltol, verbose, assumptions
        )
    else
        return nothing
    end
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::PureUMFPACKFactorization; kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        Asp = SparseMatrixCSC(
            size(A)..., getcolptr(A), rowvals(A), nonzeros(A)
        )
        # No symbolic-reuse / in-place-refactor API in PureUMFPACK: a fresh `splu`
        # recomputes ordering, symbolic analysis, and numerics together, so each
        # fresh solve refactorizes. `reuse_symbolic`/`check_pattern` are accepted
        # for API parity with `UMFPACKFactorization` but cannot skip the symbolic
        # step here. `check = false` keeps singular matrices from throwing; the
        # diagonal check below maps that to `ReturnCode.Infeasible`.
        fact = splu(Asp; check = false)
        cache.cacheval = fact
        cache.isfresh = false
    end

    F = LinearSolve.@get_cacheval(cache, :PureUMFPACKFactorization)
    # A zero on U's diagonal means a singular (or numerically singular) factor;
    # PureUMFPACK with `check = false` produces it instead of throwing.
    return if !any(iszero, diag(F.U))
        y = PureUMFPACK.solve(F, cache.b)
        copyto!(cache.u, y)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
        )
    else
        @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Infeasible
        )
    end
end

end # module
