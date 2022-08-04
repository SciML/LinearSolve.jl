module LinearSolve

using ArrayInterfaceCore
using RecursiveFactorization
using Base: cache_dependencies, Bool
import Base: eltype, adjoint, inv
using LinearAlgebra
using SparseArrays
using SciMLOperators
using SciMLOperators: IdentityOperator, InvertedOperator
using Setfield
using UnPack
using SuiteSparse
using KLU
using FastLapackInterface
using DocStringExtensions
import GPUArraysCore

# wrap
import Krylov
import KrylovKit
import IterativeSolvers

using Reexport
@reexport using SciMLBase

function isidentity(A)
    return (A === LinearAlgebra.I) |
           (A isa IterativeSolvers.Identity) |
           (A isa SciMLOperators.IdentityOperator)
end

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end
abstract type AbstractSolveFunction <: SciMLLinearSolveAlgorithm end

# Traits

needs_concrete_A(alg::AbstractFactorization) = true
needs_concrete_A(alg::AbstractKrylovSubspaceMethod) = false
needs_concrete_A(alg::AbstractSolveFunction) = false

# Code
include("common.jl")
include("factorization.jl")
include("simplelu.jl")
include("iterative_wrappers.jl")
include("solve_function.jl")
include("default.jl")
include("init.jl")

# deprecate preconditioner interface in favor of SciMLOperators
@deprecate ComposePreconditioner SciMLOperators.ComposedOperator
@deprecate InvPreconditioner SciMLOperators.InvertedOperator

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
       GenericLUFactorization, SimpleLUFactorization, RFLUFactorization,
       UMFPACKFactorization, KLUFactorization, FastLUFactorization, FastQRFactorization

export LinearSolveFunction, DirectLdiv

export KrylovJL, KrylovJL_CG, KrylovJL_GMRES, KrylovJL_BICGSTAB, KrylovJL_MINRES,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES,
       KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES

end
