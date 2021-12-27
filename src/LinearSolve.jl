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
using Requires
using SuiteSparse
using KLU
using DocStringExtensions

# wrap
import Krylov
import KrylovKit # TODO
import IterativeSolvers

using Reexport
@reexport using SciMLBase

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end

# Traits

needs_concrete_A(alg::AbstractFactorization) = true
needs_concrete_A(alg::AbstractKrylovSubspaceMethod) = true

# Code

include("common.jl")
include("factorization.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
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

  @require CUDA="052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
  @require Pardiso="46dd5b70-b6fb-5a00-ae2d-e8fea33afaf2" include("pardiso.jl")
end

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
       RFLUFactorization, UMFPACKFactorization, KLUFactorization
export KrylovJL, KrylovJL_CG, KrylovJL_GMRES, KrylovJL_BICGSTAB, KrylovJL_MINRES,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES

end
