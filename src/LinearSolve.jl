module LinearSolve

using ArrayInterface: lu_instance
using Base: cache_dependencies, Bool
using LinearAlgebra
using Reexport
using SciMLBase: AbstractDiffEqOperator, AbstractLinearAlgorithm
using Setfield
using UnPack

# wrap
using Krylov
#using IterativeSolvers

@reexport using SciMLBase

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

include("common.jl")
include("factorization.jl")
include("krylov.jl")

export LUFactorization, SVDFactorization, QRFactorization
export KrylovJL

end
