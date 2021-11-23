module LinearSolve

using ArrayInterface
using RecursiveFactorization
using Base: cache_dependencies, Bool
using LinearAlgebra
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

export LUFactorization, SVDFactorization, QRFactorization, DefaultFactorization
export KrylovJL, KrylovJL_CG, KrylovJL_GMRES, KrylovJL_BICGSTAB,
       KrylovJL_MINRES, 
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES
export DefaultLinSolve

end
