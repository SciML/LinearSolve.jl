module LinearSolveCUDSSExt

using LinearSolve: LinearSolve, cudss_loaded, LUFactorization, OperatorAssumptions, init_cacheval, LinearCache
using LinearSolve.SciMLBase
using LinearSolve.LinearAlgebra
using CUDSS
using CUDSS.CUDA.CUSPARSE: CuSparseMatrixCSR

LinearSolve.cudss_loaded(A::CuSparseMatrixCSR) = true

function LinearSolve.init_cacheval(
        ::LUFactorization, A::CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    # Return nothing - we'll create a fresh factorization each time
    # to avoid CUDSS state management issues
    nothing
end

# Custom solve! for CUDSS that creates a fresh factorization each time
function SciMLBase.solve!(cache::LinearCache{<:CuSparseMatrixCSR}, alg::LUFactorization; kwargs...)
    A = cache.A
    # Always create a fresh LU factorization for CUDSS
    fact = lu(A, check = false)
    cache.isfresh = false

    y = ldiv!(cache.u, fact, cache.b)
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

end
