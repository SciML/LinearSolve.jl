module LinearSolveCUDAExt

using CUDA
using LinearSolve
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase, LinearSolve.ArrayInterface
using SciMLBase: AbstractSciMLOperator

function LinearSolve.defaultalg(A::CUDA.CUSPARSE.CuSparseMatrixCSR{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}) where {Tv, Ti}
    if LinearSolve.cudss_loaded(A)
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
    else
        if !LinearSolve.ALREADY_WARNED_CUDSS[]
            @warn("CUDSS.jl is required for LU Factorizations on CuSparseMatrixCSR. Please load this library. Falling back to Krylov")
            LinearSolve.ALREADY_WARNED_CUDSS[] = true
        end
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES)
    end
end

function LinearSolve.error_no_cudss_lu(A::CUDA.CUSPARSE.CuSparseMatrixCSR)
    if !LinearSolve.CUDSS_LOADED[]
        error("CUDSS.jl is required for LU Factorizations on CuSparseMatrixCSR. Please load this library.")
    end
    nothing
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CudaOffloadFactorization;
        kwargs...)
    if cache.isfresh
        fact = qr(CUDA.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(CUDA.CuArray(cache.u), cache.cacheval, CUDA.CuArray(cache.b)))
    cache.u .= y
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CudaOffloadFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
        assumptions::OperatorAssumptions)
    qr(CUDA.CuArray(A))
end

function LinearSolve.init_cacheval(
        ::SparspakFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::KLUFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::UMFPACKFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Bool, assumptions::OperatorAssumptions)
    nothing
end

end
