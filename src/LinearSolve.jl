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
include("default.jl")


export LinSolveFactorize, LinSolveGPUFactorize, LinSolveLUFactorize
export LinSolveIterativeSolvers, LinSolveKrylov
export LinSolveBiCGStabl,
    LinSolveCG, LinSolveChebyshev, LinSolveGMRES, LinSolveMINRES
export DefaultLinSolve

end
