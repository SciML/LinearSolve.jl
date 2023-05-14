module LinearSolveCUDA

using CUDA, LinearAlgebra, LinearSolve, SciMLBase
using SciMLBase: AbstractSciMLOperator

struct CudaOffloadFactorization <: LinearSolve.AbstractFactorization end

function SciMLBase.solve(cache::LinearSolve.LinearCache, alg::CudaOffloadFactorization;
                         kwargs...)
    if cache.isfresh
        fact = LinearSolve.do_factorization(alg, CUDA.CuArray(cache.A), cache.b, cache.u)
        cache = LinearSolve.set_cacheval(cache, fact)
        cache.isfresh = false
    end

    copyto!(cache.u, cache.b)
    y = Array(ldiv!(cache.cacheval, CUDA.CuArray(cache.u)))
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.do_factorization(alg::CudaOffloadFactorization, A, b, u)
    A isa Union{AbstractMatrix, AbstractSciMLOperator} ||
        error("LU is not defined for $(typeof(A))")

    if A isa Union{MatrixOperator, DiffEqArrayOperator}
        A = A.A
    end

    fact = qr(CUDA.CuArray(A))
    return fact
end

export CudaOffloadFactorization

end
