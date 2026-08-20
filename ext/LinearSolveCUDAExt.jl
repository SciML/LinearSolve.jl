module LinearSolveCUDAExt

using cuSOLVER: cuSOLVER
# cuSOLVER is the only trigger, and it is what exposes the rest of the CUDA stack to
# this extension; cuSPARSE and CUDACore are not reachable on their own from here.
CUDACore = cuSOLVER.CUDACore
cuSPARSE = cuSOLVER.cuSPARSE

using LinearSolve: LinearSolve, OperatorAssumptions,
    CudaOffloadFactorization, CudaOffloadLUFactorization, CudaOffloadQRFactorization,
    CUDAOffload32MixedLUFactorization,
    SparspakFactorization, KLUFactorization, UMFPACKFactorization, LinearVerbosity
using LinearAlgebra: LinearAlgebra, LU, ldiv!, lu, qr
using SciMLBase: SciMLBase

LinearSolve.usecuda(x::Nothing) = CUDACore.functional()

function LinearSolve.is_cusparse(
        A::Union{
            cuSPARSE.CuSparseMatrixCSR, cuSPARSE.CuSparseMatrixCSC,
        }
    )
    return true
end
LinearSolve.is_cusparse_csr(::cuSPARSE.CuSparseMatrixCSR) = true
LinearSolve.is_cusparse_csc(::cuSPARSE.CuSparseMatrixCSC) = true

# CUSPARSE's COO routines require the entries sorted by row. A COO assembled by hand is
# not necessarily sorted that way (`findnz` on a CSC yields column-major order), and the
# mismatch is silent: the solve converges to a wrong answer rather than failing. The
# conversions sort as part of the conversion, so they are the way in.
# See https://github.com/SciML/LinearSolve.jl/issues/350.
function LinearSolve._check_matrix_support(::cuSPARSE.CuSparseMatrixCOO)
    return error(
        "CuSparseMatrixCOO is not supported by LinearSolve.jl. CUSPARSE requires the " *
            "entries sorted by row, which a hand-assembled COO need not be, and an " *
            "unsorted one solves to a wrong answer without erroring. Convert first, " *
            "with `CuSparseMatrixCSR(A)` or `CuSparseMatrixCSC(A)`."
    )
end

function LinearSolve.defaultalg(
        A::cuSPARSE.CuSparseMatrixCSR{Tv, Ti}, b,
        assump::OperatorAssumptions{Bool}
    ) where {Tv, Ti}
    return if LinearSolve.cudss_loaded(A)
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.LUFactorization)
    else
        if !LinearSolve.ALREADY_WARNED_CUDSS[]
            @warn("CUDSS.jl is required for LU Factorizations on CuSparseMatrixCSR. Please load this library. Falling back to Krylov")
            LinearSolve.ALREADY_WARNED_CUDSS[] = true
        end
        LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES)
    end
end

function LinearSolve.defaultalg(
        A::cuSPARSE.CuSparseMatrixCSC, b,
        assump::OperatorAssumptions{Bool}
    )
    if LinearSolve.cudss_loaded(A)
        @warn("CUDSS.jl does not support CuSparseMatrixCSC for LU Factorizations, consider using CuSparseMatrixCSR instead. Falling back to Krylov", maxlog = 1)
    else
        @warn("CuSparseMatrixCSC does not support LU Factorization falling back to Krylov. Consider using CUDSS.jl together with CuSparseMatrixCSR", maxlog = 1)
    end
    return LinearSolve.DefaultLinearSolver(LinearSolve.DefaultAlgorithmChoice.KrylovJL_GMRES)
end

function LinearSolve.error_no_cudss_lu(A::cuSPARSE.CuSparseMatrixCSR)
    if !LinearSolve.cudss_loaded(A)
        error("CUDSS.jl is required for LU Factorizations on CuSparseMatrixCSR. Please load this library.")
    end
    return nothing
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CudaOffloadLUFactorization;
        kwargs...
    )
    if cache.isfresh
        cacheval = LinearSolve.@get_cacheval(cache, :CudaOffloadLUFactorization)
        fact = lu(CUDACore.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    fact = LinearSolve.@get_cacheval(cache, :CudaOffloadLUFactorization)
    y = Array(ldiv!(CUDACore.CuArray(cache.u), fact, CUDACore.CuArray(cache.b)))
    cache.u .= y
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing)
end

function LinearSolve.init_cacheval(
        alg::CudaOffloadLUFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Check if CUDA is functional before creating CUDA arrays
    if !CUDACore.functional()
        return nothing
    end

    T = eltype(A)
    noUnitT = typeof(zero(T))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = CUDACore.CuVector{Int32}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    return LU{luT}(CUDACore.CuMatrix{Float64}(undef, 0, 0), ipiv, info)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CudaOffloadQRFactorization;
        kwargs...
    )
    if cache.isfresh
        fact = qr(CUDACore.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(CUDACore.CuArray(cache.u), cache.cacheval, CUDACore.CuArray(cache.b)))
    cache.u .= y
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing)
end

function LinearSolve.init_cacheval(
        alg::CudaOffloadQRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    # Check if CUDA is functional before creating CUDA arrays
    if !CUDACore.functional()
        return nothing
    end

    return qr(CUDACore.CuArray(A))
end

# Keep the deprecated CudaOffloadFactorization working by forwarding to QR
function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CudaOffloadFactorization;
        kwargs...
    )
    if cache.isfresh
        fact = qr(CUDACore.CuArray(cache.A))
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = Array(ldiv!(CUDACore.CuArray(cache.u), cache.cacheval, CUDACore.CuArray(cache.b)))
    cache.u .= y
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing)
end

function LinearSolve.init_cacheval(
        alg::CudaOffloadFactorization, A::AbstractArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return qr(CUDACore.CuArray(A))
end

# `qr` on a `CuSparseMatrix` falls through to the generic sparse path and dies on scalar
# indexing, so a sparse QR on the GPU was not reachable through `QRFactorization` at all.
# cuSOLVER's `csrlsvqr!` does the whole thing on the device. It wants CSR with `Int32`
# indices, which is also the layout cuSOLVER's other sparse entry points take.
# See https://github.com/SciML/LinearSolve.jl/issues/410.
function SciMLBase.solve!(
        cache::LinearSolve.LinearCache{<:cuSPARSE.CuSparseMatrixCSR{T, Int32}},
        alg::LinearSolve.QRFactorization; kwargs...
    ) where {T}
    A = cache.A
    # `csrlsvqr!` factorizes and solves in one call, so there is nothing to cache between
    # solves; `cache.isfresh` is cleared only to keep the flag honest.
    cuSOLVER.csrlsvqr!(A, cache.b, cache.u, T(cache.reltol), Cint(1), 'O')
    cache.isfresh = false
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode = LinearSolve.ReturnCode.Success
    )
end

for AlgType in (SparspakFactorization, LinearSolve.QRFactorization)
    @eval function LinearSolve.init_cacheval(
            ::$AlgType, A::cuSPARSE.CuSparseMatrixCSR, b, u,
            Pl, Pr, maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        )
        return nothing
    end
    @eval function LinearSolve.init_cacheval(
            ::$AlgType, A::cuSPARSE.CuSparseMatrixCSC, b, u,
            Pl, Pr, maxiters::Int, abstol, reltol,
            verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
        )
        return nothing
    end
end

function LinearSolve.init_cacheval(
        ::KLUFactorization, A::cuSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function LinearSolve.init_cacheval(
        ::UMFPACKFactorization, A::cuSPARSE.CuSparseMatrixCSR, b, u,
        Pl, Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

# Mixed precision CUDA LU implementation
function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::CUDAOffload32MixedLUFactorization;
        kwargs...
    )
    T32 = eltype(cache.A) <: Complex ? ComplexF32 : Float32
    if cache.isfresh
        fact, A_gpu_f32, b_gpu_f32, u_gpu_f32 = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)
        if isempty(A_gpu_f32)
            m, n = size(cache.A)
            A_gpu_f32 = CUDACore.CuMatrix{T32}(undef, m, n)
            b_gpu_f32 = CUDACore.CuVector{T32}(undef, size(cache.b, 1))
            u_gpu_f32 = CUDACore.CuVector{T32}(undef, size(cache.u, 1))
        end
        A_f32 = T32.(cache.A)
        copyto!(A_gpu_f32, A_f32)
        fact = lu(A_gpu_f32)
        cache.cacheval = (fact, A_gpu_f32, b_gpu_f32, u_gpu_f32)
        cache.isfresh = false
    end
    fact, A_gpu_f32, b_gpu_f32, u_gpu_f32 = LinearSolve.@get_cacheval(cache, :CUDAOffload32MixedLUFactorization)

    Torig = eltype(cache.u)

    # Convert b to Float32, solve, then convert back to original precision
    b_f32 = T32.(cache.b)
    copyto!(b_gpu_f32, b_f32)
    ldiv!(u_gpu_f32, fact, b_gpu_f32)
    # Convert back to original precision
    y = Array(u_gpu_f32)
    cache.u .= Torig.(y)
    return SciMLBase.build_linear_solution(alg, cache.u, nothing, nothing)
end

function LinearSolve.init_cacheval(
        alg::CUDAOffload32MixedLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    if !CUDACore.functional()
        return nothing
    end

    T32 = eltype(A) <: Complex ? ComplexF32 : Float32
    noUnitT = typeof(zero(T32))
    luT = LinearAlgebra.lutype(noUnitT)
    ipiv = CUDACore.CuVector{Int32}(undef, 0)
    info = zero(LinearAlgebra.BlasInt)
    fact = LU{luT}(CUDACore.CuMatrix{T32}(undef, 0, 0), ipiv, info)
    A_gpu_f32 = CUDACore.CuMatrix{T32}(undef, 0, 0)
    b_gpu_f32 = CUDACore.CuVector{T32}(undef, 0)
    u_gpu_f32 = CUDACore.CuVector{T32}(undef, 0)
    return (fact, A_gpu_f32, b_gpu_f32, u_gpu_f32)
end

end
