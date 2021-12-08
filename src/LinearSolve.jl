module LinearSolve

using ArrayInterface
using RecursiveFactorization
using Base: cache_dependencies, Bool
import Base: eltype, adjoint, inv
using LinearAlgebra
using IterativeSolvers:Identity
using SparseArrays
using SciMLBase: AbstractDiffEqOperator, AbstractLinearAlgorithm
using Setfield
using UnPack

# wrap
import Krylov
import KrylovKit
import IterativeSolvers

using Reexport
@reexport using SciMLBase

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end

include("common.jl")
include("factorization.jl")
include("wrappers.jl")
include("default.jl")

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

function __init__()
  @static if VERSION < v"1.7beta"
    blas = BLAS.vendor()
    IS_OPENBLAS[] = blas == :openblas64 || blas == :openblas
  else
    IS_OPENBLAS[] = occursin("openblas", BLAS.get_config().loaded_libs[1].libname)
  end
end

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
       RFLUFactorizaation
export KrylovJL, KrylovJL_CG, KrylovJL_GMRES, KrylovJL_BICGSTAB,
       KrylovJL_MINRES,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES
export DefaultLinSolve

end
