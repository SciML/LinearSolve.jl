module LinearSolveKLUExt

using LinearSolve, LinearSolve.LinearAlgebra
using KLU, KLU.SparseArrays

const PREALLOCATED_KLU = KLU.KLUFactorization(SparseMatrixCSC(0, 0, [1], Int[],
    Float64[]))

function init_cacheval(alg::KLUFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function init_cacheval(alg::KLUFactorization, A::SparseMatrixCSC{Float64, Int}, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_KLU
end

function init_cacheval(alg::KLUFactorization, A::AbstractSparseArray, b, u, Pl, Pr,
        maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    return KLU.KLUFactorization(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
        nonzeros(A)))
end

# TODO: guard this against errors
function SciMLBase.solve!(cache::LinearCache, alg::KLUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        cacheval = @get_cacheval(cache, :KLUFactorization)
        if alg.reuse_symbolic
            if alg.check_pattern && pattern_changed(cacheval, A)
                fact = KLU.klu(
                    SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                        nonzeros(A)),
                    check = false)
            else
                fact = KLU.klu!(cacheval, nonzeros(A), check = false)
            end
        else
            # New fact each time since the sparsity pattern can change
            # and thus it needs to reallocate
            fact = KLU.klu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    F = @get_cacheval(cache, :KLUFactorization)
    if F.common.status == KLU.KLU_OK
        y = ldiv!(cache.u, F, cache.b)
        SciMLBase.build_linear_solution(alg, y, nothing, cache)
    else
        SciMLBase.build_linear_solution(
            alg, cache.u, nothing, cache; retcode = ReturnCode.Infeasible)
    end
end

end
