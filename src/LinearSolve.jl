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

# OpenBLAS_jll is a standard library, but allow users to disable it via preferences
if Preferences.@load_preference("LoadOpenBLAS_JLL", true)
    using OpenBLAS_jll: OpenBLAS_jll
    const useopenblas = OpenBLAS_jll.is_available()
else
    const useopenblas = false
end

@reexport using SciMLBase

"""
    SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm

The root abstract type for all linear solver algorithms in LinearSolve.jl.
All concrete linear solver implementations should inherit from one of the
specialized subtypes rather than directly from this type.

This type integrates with the SciMLBase ecosystem, providing a consistent
interface for linear algebra operations across the Julia scientific computing
ecosystem.
"""
abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

"""
    AbstractFactorization <: SciMLLinearSolveAlgorithm

Abstract type for linear solvers that work by computing a matrix factorization.
These algorithms typically decompose the matrix `A` into a product of simpler
matrices (e.g., `A = LU`, `A = QR`, `A = LDL'`) and then solve the system
using forward/backward substitution.

## Characteristics

  - Requires concrete matrix representation (`needs_concrete_A() = true`)
  - Typically efficient for multiple solves with the same matrix
  - Generally provides high accuracy for well-conditioned problems
  - Memory requirements depend on the specific factorization type

## Subtypes

  - `AbstractDenseFactorization`: For dense matrix factorizations
  - `AbstractSparseFactorization`: For sparse matrix factorizations

## Examples of concrete subtypes

  - `LUFactorization`, `QRFactorization`, `CholeskyFactorization`
  - `UMFPACKFactorization`, `KLUFactorization`
"""
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end

"""
    AbstractSparseFactorization <: AbstractFactorization

Abstract type for factorization-based linear solvers optimized for sparse matrices.
These algorithms take advantage of sparsity patterns to reduce memory usage and
computational cost compared to dense factorizations.

## Characteristics

  - Optimized for matrices with many zero entries
  - Often use specialized pivoting strategies to preserve sparsity
  - May reorder rows/columns to minimize fill-in during factorization
  - Typically more memory-efficient than dense methods for sparse problems

## Examples of concrete subtypes

  - `UMFPACKFactorization`: General sparse LU with partial pivoting
  - `KLUFactorization`: Sparse LU optimized for circuit simulation
  - `CHOLMODFactorization`: Sparse Cholesky for positive definite systems
  - `SparspakFactorization`: Envelope/profile method for sparse systems
"""
abstract type AbstractSparseFactorization <: AbstractFactorization end

"""
    AbstractDenseFactorization <: AbstractFactorization

Abstract type for factorization-based linear solvers optimized for dense matrices.
These algorithms assume the matrix has no particular sparsity structure and use
dense linear algebra routines (typically from BLAS/LAPACK) for optimal performance.

## Characteristics

  - Optimized for matrices with few zeros or no sparsity structure
  - Leverage highly optimized BLAS/LAPACK routines when available
  - Generally provide excellent performance for moderately-sized dense problems
  - Memory usage scales as O(n²) with matrix size

## Examples of concrete subtypes

  - `LUFactorization`: Dense LU with partial pivoting (via LAPACK)
  - `QRFactorization`: Dense QR factorization for overdetermined systems
  - `CholeskyFactorization`: Dense Cholesky for symmetric positive definite matrices
  - `BunchKaufmanFactorization`: For symmetric indefinite matrices
"""
abstract type AbstractDenseFactorization <: AbstractFactorization end

"""
    AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm

Abstract type for iterative linear solvers based on Krylov subspace methods.
These algorithms solve linear systems by iteratively building an approximation
from a sequence of Krylov subspaces, without requiring explicit matrix factorization.

## Characteristics

  - Does not require concrete matrix representation (`needs_concrete_A() = false`)
  - Only needs matrix-vector products `A*v` (can work with operators/functions)
  - Memory usage typically O(n) or O(kn) where k << n
  - Convergence depends on matrix properties (condition number, eigenvalue distribution)
  - Often benefits significantly from preconditioning

## Advantages

  - Low memory requirements for large sparse systems
  - Can handle matrix-free operators (functions that compute `A*v`)
  - Often the only feasible approach for very large systems
  - Can exploit matrix structure through specialized operators

## Examples of concrete subtypes

  - `GMRESIteration`: Generalized Minimal Residual method
  - `CGIteration`: Conjugate Gradient (for symmetric positive definite systems)
  - `BiCGStabLIteration`: Bi-Conjugate Gradient Stabilized
  - Wrapped external iterative solvers (KrylovKit.jl, IterativeSolvers.jl)
"""
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end

"""
    AbstractSolveFunction <: SciMLLinearSolveAlgorithm

Abstract type for linear solvers that wrap custom solving functions or
provide direct interfaces to specific solve methods. These provide flexibility
for integrating custom algorithms or simple solve strategies.

## Characteristics

  - Does not require concrete matrix representation (`needs_concrete_A() = false`)
  - Provides maximum flexibility for custom solving strategies
  - Can wrap external solver libraries or implement specialized algorithms
  - Performance and stability depend entirely on the wrapped implementation

## Examples of concrete subtypes

  - `LinearSolveFunction`: Wraps arbitrary user-defined solve functions
  - `DirectLdiv!`: Direct application of the `\\` operator
"""
abstract type AbstractSolveFunction <: SciMLLinearSolveAlgorithm end

# Traits

"""
    needs_concrete_A(alg) -> Bool

Trait function that determines whether a linear solver algorithm requires
a concrete matrix representation or can work with abstract operators.

## Arguments

  - `alg`: A linear solver algorithm instance

## Returns

  - `true`: Algorithm requires a concrete matrix (e.g., for factorization)
  - `false`: Algorithm can work with abstract operators (e.g., matrix-free methods)

## Usage

This trait is used internally by LinearSolve.jl to optimize algorithm dispatch
and determine when matrix operators need to be converted to concrete arrays.

## Algorithm-Specific Behavior

  - `AbstractFactorization`: `true` (needs explicit matrix entries for factorization)
  - `AbstractKrylovSubspaceMethod`: `false` (only needs matrix-vector products)
  - `AbstractSolveFunction`: `false` (depends on the wrapped function's requirements)

## Example

```julia
needs_concrete_A(LUFactorization())  # true
needs_concrete_A(GMRESIteration())   # false
```
"""
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
    BLISLUFactorization
    CudaOffloadLUFactorization
    MetalLUFactorization
end

# Autotune preference constants - loaded once at package import time

# Algorithm availability checking functions
"""
    is_algorithm_available(alg::DefaultAlgorithmChoice.T)

Check if the given algorithm is currently available (extensions loaded, etc.).
"""
function is_algorithm_available(alg::DefaultAlgorithmChoice.T)
    if alg === DefaultAlgorithmChoice.LUFactorization
        return true  # Always available
    elseif alg === DefaultAlgorithmChoice.GenericLUFactorization
        return true  # Always available
    elseif alg === DefaultAlgorithmChoice.MKLLUFactorization
        return usemkl  # Available if MKL is loaded
    elseif alg === DefaultAlgorithmChoice.AppleAccelerateLUFactorization
        return appleaccelerate_isavailable()  # Available on macOS with Accelerate
    elseif alg === DefaultAlgorithmChoice.RFLUFactorization
        return userecursivefactorization(nothing)  # Requires RecursiveFactorization extension
    elseif alg === DefaultAlgorithmChoice.BLISLUFactorization
        return useblis()  # Available if BLIS extension is loaded
    elseif alg === DefaultAlgorithmChoice.CudaOffloadLUFactorization
        return usecuda()  # Available if CUDA extension is loaded
    elseif alg === DefaultAlgorithmChoice.MetalLUFactorization
        return usemetal()  # Available if Metal extension is loaded
    else
        # For extension-dependent algorithms not explicitly handled above,
        # we cannot easily check availability without trying to use them.
        # For now, assume they're not available in the default selection.
        # This includes other extensions that might be added in the future.
        return false
    end
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
include("openblas.jl")
include("simplelu.jl")
include("simplegmres.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
include("preferences.jl")
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
    :MKLLUFactorization, :MetalLUFactorization, :CUSOLVERRFFactorization)
    @eval needs_square_A(::$(alg)) = true
end

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

const HAS_APPLE_ACCELERATE = Ref(false)
appleaccelerate_isavailable() = HAS_APPLE_ACCELERATE[]

# Extension availability checking functions
useblis() = Base.get_extension(@__MODULE__, :LinearSolveBLISExt) !== nothing
usecuda() = Base.get_extension(@__MODULE__, :LinearSolveCUDAExt) !== nothing

# Metal is only available on Apple platforms
@static if !Sys.isapple()
    usemetal() = false
else
    usemetal() = Base.get_extension(@__MODULE__, :LinearSolveMetalExt) !== nothing
end

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
       BunchKaufmanFactorization, CHOLMODFactorization, LDLtFactorization,
       CUSOLVERRFFactorization, CliqueTreesFactorization

export LinearSolveFunction, DirectLdiv!, show_algorithm_choices

export KrylovJL, KrylovJL_CG, KrylovJL_MINRES, KrylovJL_GMRES,
       KrylovJL_BICGSTAB, KrylovJL_LSMR, KrylovJL_CRAIGMR,
       IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
       IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES, IterativeSolversJL_IDRS,
       KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES, KrylovJL_MINARES

export SimpleGMRES

export HYPREAlgorithm
export CudaOffloadFactorization
export CudaOffloadLUFactorization
export CudaOffloadQRFactorization
export CUDAOffload32MixedLUFactorization
export AMDGPUOffloadLUFactorization, AMDGPUOffloadQRFactorization
export MKLPardisoFactorize, MKLPardisoIterate
export PanuaPardisoFactorize, PanuaPardisoIterate
export PardisoJL
export MKLLUFactorization
export OpenBLASLUFactorization
export OpenBLAS32MixedLUFactorization
export MKL32MixedLUFactorization
export AppleAccelerateLUFactorization
export AppleAccelerate32MixedLUFactorization
export RF32MixedLUFactorization
export MetalLUFactorization
export MetalOffload32MixedLUFactorization

export OperatorAssumptions, OperatorCondition

export LinearSolveAdjoint

end
