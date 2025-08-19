module LinearSolveCUDAExt

using CUDA
using CUDA: CuVector, CuMatrix
using LinearSolve: LinearSolve, is_cusparse, defaultalg, cudss_loaded, DefaultLinearSolver,
                   DefaultAlgorithmChoice, ALREADY_WARNED_CUDSS, LinearCache,
                   needs_concrete_A,
                   error_no_cudss_lu, init_cacheval, OperatorAssumptions,
                   CudaOffloadFactorization, CudaOffloadLUFactorization, CudaOffloadQRFactorization,
                   SparspakFactorization, KLUFactorization, UMFPACKFactorization,
                   LinearVerbosity
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase, LinearSolve.ArrayInterface
using LinearAlgebra: LU
using SciMLBase: AbstractSciMLOperator

function LinearSolve.is_cusparse(A::Union{
        CUDA.CUSPARSE.CuSparseMatrixCSR, CUDA.CUSPARSE.CuSparseMatrixCSC})
    true
end

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
    if !LinearSolve.cudss_loaded(A)
        error("CUDSS.jl is required for LU Factorizations on CuSparseMatrixCSR. Please load this library.")
    end
    nothing
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CudaOffloadLUFactorization;
        kwargs...)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :CudaOffloadLUFactorization)
        fact = lu(CUDA.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    fact = LinearSolve.@get_cacheval(cache, :CudaOffloadLUFactorization)
    y = Array(ldiv!(CUDA.CuArray(cache.u), fact, CUDA.CuArray(cache.b)))
    cache.u .= y
    SciMLBase.build_linear_solution(alg, y, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CudaOffloadLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    T = eltype(A)
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = CuVector{Int32}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LU{luT}(CuMatrix{Float64}(undef, 0, 0), ipiv, info)
end

function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CudaOffloadQRFactorization;
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

function LinearSolve.init_cacheval(alg::CudaOffloadQRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    qr(CUDA.CuArray(A))
end

# Keep the deprecated CudaOffloadFactorization working by forwarding to QR
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
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    qr(CUDA.CuArray(A))
end

function LinearSolve.init_cacheval(
        ::SparspakFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::LinearVerbosity, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::KLUFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::LinearVerbosity, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::UMFPACKFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::LinearVerbosity, assumptions::OperatorAssumptions)
    nothing
end

end
