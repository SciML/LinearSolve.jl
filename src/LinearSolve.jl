module LinearSolve
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import PrecompileTools
using ArrayInterface
using RecursiveFactorization
using Base: cache_dependencies, Bool
using LinearAlgebra
using SparseArrays
using SparseArrays: AbstractSparseMatrixCSC, nonzeros, rowvals, getcolptr
using LazyArrays: @~, BroadcastArray
using SciMLBase: AbstractLinearAlgorithm, LinearAliasSpecifier
using SciMLOperators
using SciMLOperators: AbstractSciMLOperator, IdentityOperator
using Setfield
using UnPack
using KLU
using Sparspak
using FastLapackInterface
using DocStringExtensions
using EnumX
using Markdown
using ChainRulesCore
import InteractiveUtils

import StaticArraysCore: StaticArray, SVector, MVector, SMatrix, MMatrix

using LinearAlgebra: BlasInt, LU
using LinearAlgebra.LAPACK: require_one_based_indexing,
                            chkfinite, chkstride1,
                            @blasfunc, chkargsok

import GPUArraysCore
import Preferences
import ConcreteStructs: @concrete

# wrap
import Krylov
using SciMLBase
import Preferences

const CRC = ChainRulesCore

@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    if Preferences.@load_preference("LoadMKL_JLL",
        !occursin("EPYC", Sys.cpu_info()[1].model))
        using MKL_jll
        const usemkl = MKL_jll.is_available()
    else
        const usemkl = false
    end
else
    const usemkl = false
end

using Reexport
@reexport using SciMLBase
using SciMLBase: _unwrap_val

abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end
abstract type AbstractSparseFactorization <: AbstractFactorization end
abstract type AbstractDenseFactorization <: AbstractFactorization end
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end
abstract type AbstractSolveFunction <: SciMLLinearSolveAlgorithm end

# Traits

needs_concrete_A(alg::AbstractFactorization) = true
needs_concrete_A(alg::AbstractSparseFactorization) = true
needs_concrete_A(alg::AbstractKrylovSubspaceMethod) = false
needs_concrete_A(alg::AbstractSolveFunction) = false

# Util
is_underdetermined(x) = false
is_underdetermined(A::AbstractMatrix) = size(A, 1) < size(A, 2)
is_underdetermined(A::AbstractSciMLOperator) = size(A, 1) < size(A, 2)

_isidentity_struct(A) = false
_isidentity_struct(λ::Number) = isone(λ)
_isidentity_struct(A::UniformScaling) = isone(A.λ)
_isidentity_struct(::SciMLOperators.IdentityOperator) = true

# Dispatch Friendly way to check if an extension is loaded
__is_extension_loaded(::Val) = false

# Check if a sparsity pattern has changed
pattern_changed(fact, A) = false

function _fast_sym_givens! end

# Code

const INCLUDE_SPARSE = Preferences.@load_preference("include_sparse", Base.USE_GPL_LIBS)

EnumX.@enumx DefaultAlgorithmChoice begin
    LUFactorization
    QRFactorization
    DiagonalFactorization
    DirectLdiv!
    SparspakFactorization
    KLUFactorization
    UMFPACKFactorization
    KrylovJL_GMRES
    GenericLUFactorization
    RFLUFactorization
    LDLtFactorization
    BunchKaufmanFactorization
    CHOLMODFactorization
    SVDFactorization
    CholeskyFactorization
    NormalCholeskyFactorization
    AppleAccelerateLUFactorization
    MKLLUFactorization
    QRFactorizationPivoted
    KrylovJL_CRAIGMR
    KrylovJL_LSMR
end

struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice.T
end

const BLASELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

include("common.jl")
include("factorization.jl")
include("appleaccelerate.jl")
include("mkl.jl")
include("simplelu.jl")
include("simplegmres.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
include("solve_function.jl")
include("default.jl")
include("init.jl")
include("extension_algs.jl")
include("adjoint.jl")
include("deprecated.jl")

@inline function _notsuccessful(F::LinearAlgebra.QRCompactWY)
    (m, n) = size(F)
    U = view(F.factors, 1:min(m, n), 1:n)
    return any(iszero, Iterators.reverse(@view U[diagind(U)]))
end
@inline _notsuccessful(F) = hasmethod(LinearAlgebra.issuccess, (typeof(F),)) ?
                            !LinearAlgebra.issuccess(F) : false

@generated function SciMLBase.solve!(cache::LinearCache, alg::AbstractFactorization;
        kwargs...)
    quote
        if cache.isfresh
            fact = do_factorization(alg, cache.A, cache.b, cache.u)
            cache.cacheval = fact

            # If factorization was not successful, return failure. Don't reset `isfresh`
            if _notsuccessful(fact)
                return SciMLBase.build_linear_solution(
                    alg, cache.u, nothing, cache; retcode = ReturnCode.Failure)
            end

            cache.isfresh = false
        end

        y = _ldiv!(cache.u, @get_cacheval(cache, $(Meta.quot(defaultalg_symbol(alg)))),
            cache.b)
        return SciMLBase.build_linear_solution(alg, y, nothing, cache)
    end
end

@static if INCLUDE_SPARSE
    include("factorization_sparse.jl")
end

# Solver Specific Traits
## Needs Square Matrix
"""
    needs_square_A(alg)

Returns `true` if the algorithm requires a square matrix.
"""
needs_square_A(::Nothing) = false  # Linear Solve automatically will use a correct alg!
needs_square_A(alg::SciMLLinearSolveAlgorithm) = true
for alg in (:QRFactorization, :FastQRFactorization, :NormalCholeskyFactorization,
    :NormalBunchKaufmanFactorization)
    @eval needs_square_A(::$(alg)) = false
end
for kralg in (Krylov.lsmr!, Krylov.craigmr!)
    @eval needs_square_A(::KrylovJL{$(typeof(kralg))}) = false
end
for alg in (:LUFactorization, :FastLUFactorization, :SVDFactorization,
    :GenericFactorization, :GenericLUFactorization, :SimpleLUFactorization,
    :RFLUFactorization, :UMFPACKFactorization, :KLUFactorization, :SparspakFactorization,
    :DiagonalFactorization, :CholeskyFactorization, :BunchKaufmanFactorization,
    :CHOLMODFactorization, :LDLtFactorization, :AppleAccelerateLUFactorization,
    :MKLLUFactorization, :MetalLUFactorization)
    @eval needs_square_A(::$(alg)) = true
end

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

const HAS_APPLE_ACCELERATE = Ref(false)
appleaccelerate_isavailable() = HAS_APPLE_ACCELERATE[]

PrecompileTools.@compile_workload begin
    A = rand(4, 4)
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob)
    sol = solve(prob, LUFactorization())
    sol = solve(prob, RFLUFactorization())
    sol = solve(prob, KrylovJL_GMRES())
end

@static if INCLUDE_SPARSE
    PrecompileTools.@compile_workload begin
        A = sprand(4, 4, 0.3) + I
        b = rand(4)
        prob = LinearProblem(A, b)
        sol = solve(prob, KLUFactorization())
        sol = solve(prob, UMFPACKFactorization())
    end
end

PrecompileTools.@compile_workload begin
    A = sprand(4, 4, 0.3) + I
    b = rand(4)
    prob = LinearProblem(A * A', b)
    sol = solve(prob) # in case sparspak is used as default
    sol = solve(prob, SparspakFactorization())
end

ALREADY_WARNED_CUDSS = Ref{Bool}(false)
error_no_cudss_lu(A) = nothing
cudss_loaded(A) = false

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
       GenericLUFactorization, SimpleLUFactorization, RFLUFactorization,
       NormalCholeskyFactorization, NormalBunchKaufmanFactorization,
       UMFPACKFactorization, KLUFactorization, FastLUFactorization, FastQRFactorization,
       SparspakFactorization, DiagonalFactorization, CholeskyFactorization,
       BunchKaufmanFactorization, CHOLMODFactorization, LDLtFactorization

export LinearSolveFunction, DirectLdiv!

export KrylovJL, KrylovJL_CG, KrylovJL_MINRES, KrylovJL_GMRES,
       KrylovJL_BICGSTAB, KrylovJL_LSMR, KrylovJL_CRAIGMR,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES, IterativeSolversJL_IDRS,
       KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES, KrylovJL_MINARES

export SimpleGMRES

export HYPREAlgorithm
export CudaOffloadFactorization
export MKLPardisoFactorize, MKLPardisoIterate
export PanuaPardisoFactorize, PanuaPardisoIterate
export PardisoJL
export MKLLUFactorization
export AppleAccelerateLUFactorization
export MetalLUFactorization

export OperatorAssumptions, OperatorCondition

export LinearSolveAdjoint

end
