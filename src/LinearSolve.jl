module LinearSolve

using ArrayInterface: lu_instance
using Base: cache_dependencies, Bool
using LinearAlgebra
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

include("common.jl")
include("factorization.jl")
include("krylov.jl")

export LUFactorization, SVDFactorization, QRFactorization
export KrylovJL, IterativeSolversJL, KrylovKitJL 

end
