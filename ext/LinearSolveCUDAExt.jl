module LinearSolveCUDAExt

using CUDA
using CUDA: CuVector, CuMatrix
using LinearSolve: LinearSolve, is_cusparse, defaultalg, cudss_loaded, DefaultLinearSolver,
                   DefaultAlgorithmChoice, ALREADY_WARNED_CUDSS, LinearCache,
                   needs_concrete_A,
                   error_no_cudss_lu, init_cacheval, OperatorAssumptions,
                   CudaOffloadFactorization, CudaOffloadLUFactorization, CudaOffloadQRFactorization,
                   CUDAOffload32MixedLUFactorization,
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

function LinearSolve.init_cacheval(alg::CudaOffloadLUFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Bool,
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
        maxiters::Int, abstol, reltol, verbose::Bool,
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

function LinearSolve.init_cacheval(alg::CudaOffloadFactorization, A::AbstractArray, b, u, Pl, Pr,
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

# Mixed precision CUDA LU implementation
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CUDAOffload32MixedLUFactorization;
        kwargs...)
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)
        # Convert to Float32 for factorization
        A_f32 = Float32.(cache.A)
        fact = lu(CUDA.CuArray(A_f32))
        cache.cacheval = fact
        cache.isfresh = false
    end
    fact = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)
    # Convert b to Float32, solve, then convert back to original precision
    b_f32 = Float32.(cache.b)
    u_f32 = CUDA.CuArray(b_f32)
    y_f32 = ldiv!(u_f32, fact, CUDA.CuArray(b_f32))
    # Convert back to original precision
    y = Array(y_f32)
    T = eltype(cache.u)
    cache.u .= T.(y)
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CUDAOffload32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::LinearVerbosity,
        assumptions::OperatorAssumptions)
    # Pre-allocate with Float32 arrays
    A_f32 = Float32.(A)
    T = eltype(A_f32)
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = CuVector{Int32}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LU{luT}(CuMatrix{Float32}(undef, 0, 0), ipiv, info)
end

end
