module LinearSolve
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end
using ArrayInterfaceCore
using RecursiveFactorization
using Base: cache_dependencies, Bool
import Base: eltype, adjoint, inv
using LinearAlgebra
using IterativeSolvers: Identity
using SparseArrays
using SciMLBase: AbstractLinearAlgorithm, DiffEqIdentity
using SciMLOperators: AbstractSciMLOperator, IdentityOperator,
                      InvertedOperator, ComposedOperator
using Setfield
using UnPack
using SuiteSparse
using KLU
using Sparspak
using FastLapackInterface
using DocStringExtensions
import GPUArraysCore
import Preferences
import SciMLOperators: issquare

# wrap
import Krylov
import KrylovKit
import IterativeSolvers

using Reexport
@reexport using SciMLBase
using SciMLBase: _unwrap_val

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end
abstract type AbstractSolveFunction <: SciMLLinearSolveAlgorithm end

# Traits

needs_concrete_A(alg::AbstractFactorization) = true
needs_concrete_A(alg::AbstractKrylovSubspaceMethod) = false
needs_concrete_A(alg::AbstractSolveFunction) = false

# Util
_isidentity_struct(A) = false
_isidentity_struct(λ::Number) = isone(λ)
_isidentity_struct(A::UniformScaling) = isone(A.λ)
_isidentity_struct(::IterativeSolvers.Identity) = true
_isidentity_struct(::SciMLBase.IdentityOperator) = true
_isidentity_struct(::SciMLBase.DiffEqIdentity) = true

# Code

const INCLUDE_SPARSE = Preferences.@load_preference("include_sparse", Base.USE_GPL_LIBS)

include("common.jl")
include("factorization.jl")
include("simplelu.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
include("solve_function.jl")
include("default.jl")
include("init.jl")
include("HYPRE.jl")

@static if INCLUDE_SPARSE
    include("factorization_sparse.jl")
end

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

import SnoopPrecompile

SnoopPrecompile.@precompile_all_calls begin
    A = rand(4, 4)
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob)
    sol = solve(prob, LUFactorization())
    sol = solve(prob, RFLUFactorization())
    sol = solve(prob, KrylovJL_GMRES())
end

@static if INCLUDE_SPARSE
    SnoopPrecompile.@precompile_all_calls begin
        A = sprand(4, 4, 0.3) + I
        b = rand(4)
        prob = LinearProblem(A, b)
        sol = solve(prob, KLUFactorization())
        sol = solve(prob, UMFPACKFactorization())
    end
end

SnoopPrecompile.@precompile_all_calls begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob) # in case sparspak is used as default
    sol = solve(prob, SparspakFactorization())
end

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
       GenericLUFactorization, SimpleLUFactorization, RFLUFactorization,
       UMFPACKFactorization, KLUFactorization, FastLUFactorization, FastQRFactorization,
       SparspakFactorization, DiagonalFactorization

export LinearSolveFunction

export KrylovJL, KrylovJL_CG, KrylovJL_MINRES, KrylovJL_GMRES,
       KrylovJL_BICGSTAB, KrylovJL_LSMR, KrylovJL_CRAIGMR,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES,
       KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES

export HYPREAlgorithm

end
