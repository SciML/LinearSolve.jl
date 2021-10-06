module LinearSolve

using ArrayInterface: lu_instance
using Base: cache_dependencies, Bool
using Krylov
using LinearAlgebra
using Reexport
using SciMLBase: AbstractDiffEqOperator, AbstractLinearAlgorithm
using Setfield
using UnPack

@reexport using SciMLBase

abstract type SciMLLinearSolveAlgorithm end

include("common.jl")
include("factorization.jl")
include("krylov.jl")

export LUFactorization, SVDFactorization, QRFactorization
export KrylovJL

end
