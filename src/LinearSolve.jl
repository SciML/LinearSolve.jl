module LinearSolve

using ArrayInterface
using ArrayInterface: lu_instance
using Base: cache_dependencies, Bool
using FastBroadcast
using IterativeSolvers
using Krylov
using LinearAlgebra
using Reexport
using Requires
using SciMLBase: AbstractDiffEqOperator, AbstractLinearAlgorithm
using Setfield
using SparseArrays
using SuiteSparse
using UnPack

@reexport using SciMLBase

cuify(x) = error("To use LinSolveGPUFactorize, you must do `using CUDA`")

is_cuda_available() = false

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" begin
        cuify(x::AbstractArray) = CUDA.CuArray(x)

        is_cuda_available() = true

        function LinearAlgebra.ldiv!(
            x::CUDA.CuArray,
            _qr::CUDA.CUSOLVER.CuQR,
            b::CUDA.CuArray,
        )
            _x = UpperTriangular(_qr.R) \ (_qr.Q' * reshape(b, length(b), 1))
            x .= vec(_x)
            CUDA.unsafe_free!(_x)
            return x
        end
        # make `\` work
        LinearAlgebra.ldiv!(F::CUDA.CUSOLVER.CuQR, b::CUDA.CuArray) =
            (x = similar(b); ldiv!(x, F, b); x)

        default_factorize(A::CUDA.CuArray) = qr(A)
    end
end

function isopenblas()
    @static if VERSION < v"1.7beta"
        blas = BLAS.vendor()
        blas == :openblas64 || blas == :openblas
    else
        occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
    end
end


include("arrays.jl")
include("factorization.jl")
include("wrappers.jl")


export LinSolveFactorize, LinSolveGPUFactorize, LinSolveLUFactorize
export LinSolveIterativeSolvers, LinSolveKrylov
export LinSolveBiCGStabl,
    LinSolveCG, LinSolveChebyshev, LinSolveGMRES, LinSolveMINRES

# export LUFactorization, SVDFactorization, QRFactorization
# export KrylovJL

end
