module LinearSolveCUDAExt

using CUDA
using LinearSolve: LinearSolve, is_cusparse, defaultalg, cudss_loaded, DefaultLinearSolver,
                   DefaultAlgorithmChoice, ALREADY_WARNED_CUDSS, LinearCache,
                   needs_concrete_A,
                   error_no_cudss_lu, init_cacheval, OperatorAssumptions,
                   CudaOffloadFactorization, CudaOffloadLUFactorization, CudaOffloadQRFactorization,
                   CUDAOffload32MixedLUFactorization,
                   SparspakFactorization, KLUFactorization, UMFPACKFactorization, LinearVerbosity
using LinearSolve.LinearAlgebra, LinearSolve.SciMLBase, LinearSolve.ArrayInterface
using SciMLBase: AbstractSciMLOperator

LinearSolve.usecuda(x::Nothing) = CUDA.functional()

function LinearSolve.is_cusparse(A::Union{
        CUDA.CUSPARSE.CuSparseMatrixCSR, CUDA.CUSPARSE.CuSparseMatrixCSC})
    true
end
LinearSolve.is_cusparse_csr(::CUDA.CUSPARSE.CuSparseMatrixCSR) = true
LinearSolve.is_cusparse_csc(::CUDA.CUSPARSE.CuSparseMatrixCSC) = true

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
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    # Check if CUDA is functional before creating CUDA arrays
    if !CUDA.functional()
        return nothing
    end
    
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
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    # Check if CUDA is functional before creating CUDA arrays
    if !CUDA.functional()
        return nothing
    end
    
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
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    qr(CUDA.CuArray(A))
end

function LinearSolve.init_cacheval(
        ::SparspakFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::KLUFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

function LinearSolve.init_cacheval(
        ::UMFPACKFactorization, A::CUDA.CUSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions)
    nothing
end

# Mixed precision CUDA LU implementation
function SciMLBase.solve!(cache::LinearSolve.LinearCache, alg::CUDAOffload32MixedLUFactorization;
        kwargs...)
    if cache.isfresh
        fact, A_gpu_f32, b_gpu_f32, u_gpu_f32 = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)
        # Compute 32-bit type on demand and convert
        T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
        A_f32 = T32.(cache.A)
        copyto!(A_gpu_f32, A_f32)
        fact = lu(A_gpu_f32)
        cache.cacheval = (fact, A_gpu_f32, b_gpu_f32, u_gpu_f32)
        cache.isfresh = false
    end
    fact, A_gpu_f32, b_gpu_f32, u_gpu_f32 = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)
    
    # Compute types on demand for conversions
    T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
    Torig = eltype(cache.u)
    
    # Convert b to Float32, solve, then convert back to original precision
    b_f32 = T32.(cache.b)
    copyto!(b_gpu_f32, b_f32)
    ldiv!(u_gpu_f32, fact, b_gpu_f32)
    # Convert back to original precision
    y = Array(u_gpu_f32)
    cache.u .= Torig.(y)
    SciMLBase.build_linear_solution(alg, cache.u, nothing, cache)
end

function LinearSolve.init_cacheval(alg::CUDAOffload32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions)
    # Pre-allocate with Float32 arrays
    m, n = size(A)
    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    noUnitT = typeof(zero(T32))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = CuVector{Int32}(undef, min(m, n))
    info = zero(LinearAlgebra.BlasInt)
    fact = LU{luT}(CuMatrix{T32}(undef, m, n), ipiv, info)
    A_gpu_f32 = CuMatrix{T32}(undef, m, n)
    b_gpu_f32 = CuVector{T32}(undef, size(b, 1))
    u_gpu_f32 = CuVector{T32}(undef, size(u, 1))
    return (fact, A_gpu_f32, b_gpu_f32, u_gpu_f32)
end

end
