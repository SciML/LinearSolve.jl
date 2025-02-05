module LinearSolveSparspakExt

using LinearSolve, LinearAlgebra
using Sparspak
using Sparspak.SparseCSCInterface.SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, getcolptr

const PREALLOCATED_SPARSEPAK = sparspaklu(SparseMatrixCSC(0, 0, [1], Int[], Float64[]),
    factorize = false)

function LinearSolve.init_cacheval(
        ::SparspakFactorization, A::SparseMatrixCSC{Float64, Int}, b, u, Pl,
        Pr, maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    PREALLOCATED_SPARSEPAK
end

function LinearSolve.init_cacheval(
        ::SparspakFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    A = convert(AbstractMatrix, A)
    if A isa SparseArrays.AbstractSparseArray
        return sparspaklu(
            SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)),
            factorize = false)
    else
        return sparspaklu(SparseMatrixCSC(0, 0, [1], Int[], eltype(A)[]),
            factorize = false)
    end
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::SparspakFactorization; kwargs...)
    A = cache.A
    if cache.isfresh
        if cache.cacheval !== nothing && alg.reuse_symbolic
            fact = sparspaklu!(LinearSolve.@get_cacheval(cache, :SparspakFactorization),
                SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                    nonzeros(A)))
        else
            fact = sparspaklu(SparseMatrixCSC(size(A)..., getcolptr(A), rowvals(A),
                nonzeros(A)))
        end
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, LinearSolve.@get_cacheval(cache, :SparspakFactorization), cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

LinearSolve.PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A * A', b)
    sol = solve(prob) # in case sparspak is used as default
    sol = solve(prob, SparspakFactorization())
end

end
