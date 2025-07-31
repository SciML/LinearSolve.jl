module LinearSolve
if isdefined(Base, :Experimental) &&
   isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import PrecompileTools
using ArrayInterface: ArrayInterface
using Base: Bool, convert, copyto!, adjoint, transpose, /, \, require_one_based_indexing
using LinearAlgebra: LinearAlgebra, BlasInt, LU, Adjoint, BLAS, Bidiagonal, BunchKaufman,
                     ColumnNorm, Diagonal, Factorization, Hermitian, I, LAPACK, NoPivot,
                     RowMaximum, RowNonZero, SymTridiagonal, Symmetric, Transpose,
                     Tridiagonal, UniformScaling, axpby!, axpy!, bunchkaufman,
                     bunchkaufman!,
                     cholesky, cholesky!, diagind, dot, inv, ldiv!, ldlt!, lu, lu!, mul!,
                     norm,
                     qr, qr!, svd, svd!
using LazyArrays: @~, BroadcastArray
using SciMLBase: SciMLBase, LinearAliasSpecifier, AbstractSciMLOperator,
                 init, solve!, reinit!, solve, ReturnCode, LinearProblem
using SciMLOperators: SciMLOperators, AbstractSciMLOperator, IdentityOperator,
                      MatrixOperator,
                      has_ldiv!, issquare
using Setfield: @set, @set!
using UnPack: @unpack
using DocStringExtensions: DocStringExtensions
using EnumX: EnumX, @enumx
using Markdown: Markdown, @doc_str
using ChainRulesCore: ChainRulesCore, NoTangent
using Reexport: Reexport, @reexport
using Libdl: Libdl, dlsym_e
import InteractiveUtils
import RecursiveArrayTools

import StaticArraysCore: StaticArray, SVector, SMatrix

using LinearAlgebra.LAPACK: chkfinite, chkstride1,
                            @blasfunc, chkargsok

import GPUArraysCore
import Preferences
import ConcreteStructs: @concrete

# wrap
import Krylov

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

@reexport using SciMLBase

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

issparsematrixcsc(A) = false
handle_sparsematrixcsc_lu(A) = lu(A)
issparsematrix(A) = false
make_SparseMatrixCSC(A) = nothing
makeempty_SparseMatrixCSC(A) = nothing

# Stub functions for SparseArrays - overridden in extension
getcolptr(A) = error("SparseArrays extension not loaded")
rowvals(A) = error("SparseArrays extension not loaded")
nonzeros(A) = error("SparseArrays extension not loaded")

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

"""
    DefaultLinearSolver(;safetyfallback=true)

The default linear solver. This is the algorithm chosen when `solve(prob)`
is called. It's a polyalgorithm that detects the optimal method for a given
`A, b` and hardware (Intel, AMD, GPU, etc.).

## Keyword Arguments

  - `safetyfallback`: determines whether to fallback to a column-pivoted QR factorization
    when an LU factorization fails. This can be required if `A` is rank-deficient. Defaults
    to true.
"""
struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice.T
    safetyfallback::Bool
    DefaultLinearSolver(alg; safetyfallback = true) = new(alg, safetyfallback)
end

const BLASELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

function defaultalg_symbol end

include("generic_lufact.jl")
include("common.jl")
include("extension_algs.jl")
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
include("adjoint.jl")

## Deprecated, remove in July 2025

@static if isdefined(SciMLBase, :DiffEqArrayOperator)
    function defaultalg(A::SciMLBase.DiffEqArrayOperator, b,
            assump::OperatorAssumptions{Bool})
        defaultalg(A.A, b, assump)
    end
end

@inline function _notsuccessful(F::LinearAlgebra.QRCompactWY{
        T, A}) where {T, A <: GPUArraysCore.AnyGPUArray}
    hasmethod(LinearAlgebra.issuccess, (typeof(F),)) ?
    !LinearAlgebra.issuccess(F) : false
end

@inline function _notsuccessful(F::LinearAlgebra.QRCompactWY)
    (m, n) = size(F)
    U = view(F.factors, 1:min(m, n), 1:n)
    return any(iszero, Iterators.reverse(@view U[diagind(U)]))
end
@inline _notsuccessful(F) = hasmethod(LinearAlgebra.issuccess, (typeof(F),)) ?
                            !LinearAlgebra.issuccess(F) : false

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
    sol = solve(prob, KrylovJL_GMRES())
end

ALREADY_WARNED_CUDSS = Ref{Bool}(false)
error_no_cudss_lu(A) = nothing
cudss_loaded(A) = false
is_cusparse(A) = false

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
