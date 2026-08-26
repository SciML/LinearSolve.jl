module LinearSolve
if isdefined(Base, :Experimental) &&
        isdefined(Base.Experimental, Symbol("@max_methods"))
    @eval Base.Experimental.@max_methods 1
end

import PrecompileTools
using ArrayInterface: ArrayInterface
# Explicit names, not the module: LinearSolve defines its own `LHLFactorization` (the
# algorithm object) and `using LHLFactorization` would shadow it.
using LHLFactorization: LHLWorkspace, lhl_reduce!, lhl_shift!, lhl_ldiv!, lhl_refine!
using Base: Bool, convert, copyto!, adjoint, transpose, /, \, require_one_based_indexing
using LinearAlgebra: LinearAlgebra, BlasInt, LU, Adjoint, BLAS, Bidiagonal, BunchKaufman,
    ColumnNorm, cond, Diagonal, Factorization, Hermitian, I, LAPACK, NoPivot,
    RowMaximum, RowNonZero, SymTridiagonal, Symmetric, Transpose,
    Tridiagonal, UniformScaling, axpby!, axpy!, bunchkaufman,
    bunchkaufman!,
    cholesky, cholesky!, diagind, dot, inv, issuccess, ldiv!, ldlt!, lu, lu!, mul!,
    norm,
    qr, qr!, svd, svd!
using SciMLBase: SciMLBase, LinearAliasSpecifier,
    init, solve!, reinit!, solve, ReturnCode, LinearProblem
using SciMLOperators: SciMLOperators, AbstractSciMLOperator, IdentityOperator,
    MatrixOperator, WOperator, jacobian_stale,
    mark_jacobian_current!,
    has_ldiv!, issquare
using SciMLStructures: SciMLStructures
using SciMLLogging: SciMLLogging, @SciMLMessage, verbosity_to_int,
    AbstractVerbositySpecifier, AbstractVerbosityPreset,
    Silent, InfoLevel, WarnLevel, MessageLevel, None, Minimal, Standard, Detailed, All
using Setfield: Setfield, @set!
using DocStringExtensions: DocStringExtensions
using EnumX: EnumX
using Markdown: Markdown
using Reexport: Reexport, @reexport
using Libdl: Libdl
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

if Int === Int64 && !Base.USE_BLAS64
    error(
        "Invalid installation of Julia detected.\n\n Detected that Julia was built in 64-bit version but with a 32-bit BLAS. This gives issues" *
            " in LinearAlgebra.jl and LinearSolve.jl which can be unrecoverable and are thus not supported. Most likely this is due to a bad build" *
            " of Julia, with the common reasons being an incorrect build script in the NixOS and ArchLinux package managers (and old versions of homebrew)." *
            " To fix this issue, and many other potentially small issues that may be undetected, use get a valid version of Julia with the correct BLAS and" *
            " LLVM versions by either installing via juliaup (recommended), or downloading the appropriate binary from https://julialang.org/install/" *
            " If using a Unix machine with a bash terminal, `curl -fsSL https://install.julialang.org` | sh will install juliaup and `juliaup add latest` will" *
            " then give the latest version.\n\n If you wish to help fix the incorrect package manager build, share the discussion on fixing the homebrew build" *
            " https://github.com/Homebrew/homebrew-core/issues/246702 with the package manager of interest in order to improve the ecosystem."
    )
end

@static if Sys.ARCH === :x86_64 || Sys.ARCH === :i686
    if Preferences.@load_preference(
            "LoadMKL_JLL",
            !occursin("EPYC", Sys.cpu_info()[1].model)
        )
        # MKL_jll < 2022.2 doesn't support the mixed LP64 and ILP64 interfaces that we make use of in LinearSolve
        # In particular, the `_64` APIs do not exist
        # https://www.intel.com/content/www/us/en/developer/articles/release-notes/onemkl-release-notes-2022.html
        using MKL_jll: MKL_jll
        const usemkl = MKL_jll.is_available() && pkgversion(MKL_jll) >= v"2022.2"
    else
        const usemkl = false
    end
else
    const usemkl = false
end

@static if usemkl
    using MKL_jll: libmkl_rt
else
    global libmkl_rt
    nothing
end

# OpenBLAS_jll is a standard library, but allow users to disable it via preferences
if Preferences.@load_preference("LoadOpenBLAS_JLL", true)
    using OpenBLAS_jll: OpenBLAS_jll, libopenblas
    const useopenblas = OpenBLAS_jll.is_available()
else
    const useopenblas = false
    global libopenblas
    nothing
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

# Interface

A concrete `MyAlg <: SciMLLinearSolveAlgorithm` must implement
`SciMLBase.solve!(cache::LinearCache, alg::MyAlg; kwargs...)` and
[`needs_concrete_A`](@ref)`(alg::MyAlg)::Bool`. It may implement
[`init_cacheval`](@ref), [`default_alias_A`](@ref), [`default_alias_b`](@ref),
[`needs_square_A`](@ref), and [`update_tolerances_internal!`](@ref); each of
these has a documented default.

Subtyping one of the categorized abstract types below supplies
`needs_concrete_A` and the aliasing defaults. Direct subtypes must define
`needs_concrete_A` themselves. The complete contract, including cache lifecycle
rules and extension boundaries, is documented on the
[Linear Solver Algorithm Interface](@ref) page.

# Extension rules

Define the four traits `needs_concrete_A`, `needs_square_A`, `default_alias_A`,
and `default_alias_b` in the package that defines `MyAlg`, next to the algorithm
type. Downstream solvers query them before an optional backend is loaded. Methods
that actually call the backend, such as `solve!` and `init_cacheval`, may be
defined in the package extension instead.

# Examples

```julia
struct MyAlg <: LinearSolve.AbstractKrylovSubspaceMethod end
LinearSolve.needs_square_A(::MyAlg) = true
```

Use [`algorithm_interface_issues`](@ref) to check a custom algorithm before
passing it to `init` or `solve`.
"""
abstract type SciMLLinearSolveAlgorithm <: SciMLBase.AbstractLinearAlgorithm end

"""
    AbstractFactorization <: SciMLLinearSolveAlgorithm

Abstract type for linear solvers that work by computing a matrix factorization.
These algorithms typically decompose the matrix `A` into a product of simpler
matrices (e.g., `A = LU`, `A = QR`, `A = LDL'`) and then solve the system
using forward/backward substitution.

# Interface

Factorization algorithms receive a concrete representation of `A`, normally
factorize it when `cache.isfresh` is true, store the factorization in
`cache.cacheval`, and return a `SciMLBase.build_linear_solution` from `solve!`.
The concrete subtypes are [`AbstractDenseFactorization`](@ref) and
[`AbstractSparseFactorization`](@ref).

# Examples

`LUFactorization`, `QRFactorization`, `CholeskyFactorization`, `UMFPACKFactorization`,
and `KLUFactorization` are concrete subtypes.
"""
abstract type AbstractFactorization <: SciMLLinearSolveAlgorithm end

"""
    AbstractSparseFactorization <: AbstractFactorization

Abstract type for factorization-based linear solvers optimized for sparse matrices.
These algorithms take advantage of sparsity patterns to reduce memory usage and
computational cost compared to dense factorizations.

# Interface

Use this supertype when the algorithm can preserve and exploit sparse structure.
The default `needs_concrete_A`, `default_alias_A`, and `default_alias_b` traits
are appropriate for the usual non-mutating sparse factorization workflow. An
algorithm that mutates its input must override the aliasing traits and document
that requirement.

# Examples

`UMFPACKFactorization`, `KLUFactorization`, `CHOLMODFactorization`,
`SparspakFactorization`, and `ParUFactorization` are concrete subtypes.
"""
abstract type AbstractSparseFactorization <: AbstractFactorization end

"""
    AbstractDenseFactorization <: AbstractFactorization

Abstract type for factorization-based linear solvers optimized for dense matrices.
These algorithms assume the matrix has no particular sparsity structure and use
dense linear algebra routines (typically from BLAS/LAPACK) for optimal performance.

# Interface

Use this supertype when the algorithm needs the entries of a dense `A` to build
a factorization. The default `needs_concrete_A` is `true` and the default
aliasing traits are `false`, preserving the caller's matrix while a
factorization is built.

# Examples

`LUFactorization`, `QRFactorization`, `CholeskyFactorization`, and
`BunchKaufmanFactorization` are concrete subtypes.
"""
abstract type AbstractDenseFactorization <: AbstractFactorization end

"""
    AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm

Abstract type for iterative linear solvers based on Krylov subspace methods.
These algorithms solve linear systems by iteratively building an approximation
from a sequence of Krylov subspaces, without requiring explicit matrix factorization.

# Interface

Use this supertype for algorithms that can solve from matrix-vector products
without materializing `A`. `needs_concrete_A` defaults to `false`. The
implementation reads the right-hand side and tolerances from `LinearCache`,
uses `cache.Pl` and `cache.Pr` when it supports preconditioning, writes into
`cache.u`, and returns a `SciMLBase.build_linear_solution`.

# Examples

Krylov wrappers such as `KrylovJL_GMRES`, `KrylovJL_CG`, and
`IterativeSolversJL_GMRES` are concrete subtypes. A matrix-free implementation
must support the `mul!` operations required by its algorithm.
"""
abstract type AbstractKrylovSubspaceMethod <: SciMLLinearSolveAlgorithm end

"""
    AbstractSolveFunction <: SciMLLinearSolveAlgorithm

Abstract type for linear solvers that wrap custom solving functions or
provide direct interfaces to specific solve methods. These provide flexibility
for integrating custom algorithms or simple solve strategies.

# Interface

Use this supertype for an algorithm that delegates solving to a callable or a
specialized direct operation. `needs_concrete_A` defaults to `false`, but the
wrapped implementation is responsible for accepting the operator types that
the algorithm advertises. The callable must return a solution compatible with
`cache.u`.

# Examples

`LinearSolveFunction` wraps a user callable, while `DirectLdiv!` delegates to
Julia's `ldiv!` implementation.
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
and determine when matrix operators need to be converted to concrete arrays. It
is also queried by downstream solvers such as OrdinaryDiffEq.jl and
NonlinearSolve.jl to decide whether to assemble a concrete Jacobian, which is
why every algorithm must implement it, and why it must be implemented next to
the algorithm struct rather than in a package extension: the callers run before
the backend package is necessarily loaded.

## Algorithm-Specific Behavior

  - `AbstractFactorization`: `true` (needs explicit matrix entries for factorization)
  - `AbstractKrylovSubspaceMethod`: `false` (only needs matrix-vector products)
  - `AbstractSolveFunction`: `false` (depends on the wrapped function's requirements)
  - Direct subtypes of `SciMLLinearSolveAlgorithm`: no default; defining this is required

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
# `SciMLOperators.has_concretization(::AbstractWOperator)` is an unconditional `true`, so a
# `WOperator` over a matrix-free Jacobian says it can be concretized and then throws from
# `convert`. The default solver builds a cacheval for every slot it holds, several of which
# concretize, so that lands as a `MethodError` from `init` before any algorithm runs. Ask
# the Jacobian instead. See https://github.com/SciML/LinearSolve.jl/issues/1236.
_has_concretization(A) = SciMLOperators.has_concretization(A)
_has_concretization(W::SciMLOperators.AbstractWOperator) = _has_concretization(W.J)

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

EnumX.@enumx DefaultAlgorithmChoice begin
    LUFactorization
    QRFactorization
    DiagonalFactorization
    DirectLdiv!
    SparspakFactorization
    KLUFactorization
    SupernodalLUFactorization
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
    SparseColumnPivotedQRFactorization
    LHLFactorization
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
        return useblis(nothing)  # Available if BLIS extension is loaded
    elseif alg === DefaultAlgorithmChoice.CudaOffloadLUFactorization
        return usecuda(nothing)  # Available if CUDA extension is loaded
    elseif alg === DefaultAlgorithmChoice.MetalLUFactorization
        return usemetal(nothing)  # Available if Metal extension is loaded
    elseif alg === DefaultAlgorithmChoice.SparseColumnPivotedQRFactorization
        return true  # SparseColumnPivotedQR is a hard dependency, always available
    elseif alg === DefaultAlgorithmChoice.LHLFactorization
        return true  # LHLFactorization.jl is a hard dependency, always available
    else
        # For extension-dependent algorithms not explicitly handled above,
        # we cannot easily check availability without trying to use them.
        # For now, assume they're not available in the default selection.
        # This includes other extensions that might be added in the future.
        return false
    end
end

"""
    DefaultLinearSolver(;safetyfallback=true, residualsafety=false)

The default linear solver. This is the algorithm chosen when `solve(prob)`
is called. It's a polyalgorithm that detects the optimal method for a given
`A, b` and hardware (Intel, AMD, GPU, etc.).

## Keyword Arguments

  - `safetyfallback`: determines whether to fallback to a column-pivoted QR factorization
    when an LU factorization fails (zero pivot) or produces non-finite values (NaN/Inf
    from near-singular matrices). Defaults to `true`.
  - `residualsafety`: when `true`, the inner LU algorithm computes the post-solve residual
    `‖A*x - b‖` and returns `ReturnCode.APosterioriSafetyFailure` if it exceeds
    `abstol + reltol * ‖b‖`. The default solver then falls back to column-pivoted QR.
    Defaults to `false`. Note: for ill-conditioned matrices, LU with partial pivoting
    always achieves optimal backward error (≈ eps), so the large forward residual reflects
    the problem conditioning, not algorithm failure. Enabling this check can trigger
    unnecessary fallbacks; callers should set appropriate `abstol`/`reltol` values.

## Residual Safety

Individual LU algorithms (e.g. `LUFactorization`, `GenericLUFactorization`, etc.) support a
`residualsafety` keyword argument. When `residualsafety=true`, the algorithm computes the
post-solve residual `‖A*x - b‖` and returns `ReturnCode.APosterioriSafetyFailure` if it
exceeds `abstol + reltol * ‖b‖`. When used standalone (not through `DefaultLinearSolver`),
no QR fallback is performed — the caller receives the failure retcode and can decide how to
handle it.
"""
struct DefaultLinearSolver <: SciMLLinearSolveAlgorithm
    alg::DefaultAlgorithmChoice.T
    safetyfallback::Bool
    residualsafety::Bool
    DefaultLinearSolver(alg; safetyfallback = true, residualsafety = false) = new(alg, safetyfallback, residualsafety)
end

const BLASELTYPES = Union{Float32, Float64, ComplexF32, ComplexF64}

function defaultalg_symbol end

"""
    _check_matrix_support(A)

Reject a matrix format that would otherwise be solved incorrectly. Extensions add
methods for their own formats; the fallback accepts everything.
"""
_check_matrix_support(A) = nothing

include("verbosity.jl")
include("blas_logging.jl")
include("generic_lufact.jl")
include("blocked_lufact.jl")
include("eigenvalue.jl")
include("common.jl")
include("interface.jl")
include("extension_algs.jl")
include("factorization.jl")
include("appleaccelerate.jl")
include("mkl.jl")
include("openblas.jl")
include("simplelu.jl")
include("lowrank.jl")
include("lhl.jl")
include("adjoint_factorization.jl")
include("simplegmres.jl")
include("iterative_wrappers.jl")
include("preconditioners.jl")
include("preferences.jl")
include("solve_function.jl")
include("default.jl")
"""
    supernodal_panel_solve!(W, B, np; operation, algorithm = :auto)

Apply a supernodal triangular-panel operation using the requested backend.

# Arguments

  - `W`: Matrix containing the factored diagonal block.
  - `B`: Panel or right-hand side to update in place.
  - `np`: Width of the diagonal block in `W`.

# Keywords

  - `operation`: which triangular operation to apply. One of

      + `:factor_right_upper`: `B := B / U11`, a right solve with the non-unit
        upper triangle (used during numeric factorization to form `L21`).
      + `:factor_lower`: `B := L11 \\ B`, a left solve with the unit-lower triangle
        (used during numeric factorization to form `U12`).
      + `:lower`: `B := L11 \\ B`, the unit-lower forward substitution of the solve
        phase.
      + `:upper`: `B := U11 \\ B`, the non-unit upper back substitution of the
        solve phase.

    Any other symbol throws an `ArgumentError`. `:factor_lower` and `:lower`
    compute the same result; they differ only in which backend code path each
    `algorithm` routes them to (see below).
  - `algorithm`: backend to dispatch to through `supernodal_panel_solve_backend!`.
    Defaults to `:auto`. Accepted values:

      + `:kernel`: the in-tree column-oriented kernels (generic over the element
        type, allocation-free) for `:lower`/`:upper`; the `:factor_*` operations
        are forwarded to `:triangularsolve`.
      + `:blas`: `BLAS.trsm!` for `:lower`/`:upper` when `W` and `B` are strided
        `BlasFloat` matrices, the kernels for other element types; the `:factor_*`
        operations are forwarded to `:triangularsolve`.
      + `:triangularsolve`: `LinearAlgebra.ldiv!`/`rdiv!` on the triangular
        wrappers, replaced by TriangularSolve.jl kernels for strided `Float32`
        and `Float64` panels when RecursiveFactorization.jl and TriangularSolve.jl
        are loaded.
      + `:auto`: `:factor_right_upper` and `:factor_lower` always go to
        `:triangularsolve`. `:lower` and `:upper` use `:kernel` when
        `np <= PANEL_KERNEL_MAX_NP` (256), when `B` has a single column, or when
        the element type is not a `BlasFloat`; otherwise `:blas` when
        `np > PANEL_BLAS_MIN_NP` (1792) and `:triangularsolve` in between.

# Returns

The updated `B`.

!!! note

    The TriangularSolve.jl backend is only active when both RecursiveFactorization.jl
    and TriangularSolve.jl are loaded; `using RecursiveFactorization` is enough,
    since RecursiveFactorization.jl depends on TriangularSolve.jl. Without them
    `:triangularsolve` routes `:lower`/`:upper` back to `:blas` and the
    `:factor_*` operations to the stdlib triangular solves.
"""
function supernodal_panel_solve! end

"""
    supernodal_panel_solve_backend!(algorithm, W, B, np; operation)

Backend extension hook for [`supernodal_panel_solve!`](@ref).

# Interface rules

An extension may specialize `algorithm::Val` for supported operand types. The method
must apply `operation` to `B` in place, return `B`, and treat the factored diagonal
block `W[1:np, 1:np]` as read-only. `B` may be a view into the remainder of `W`.
Unsupported operations must throw `ArgumentError`.

The built-in backends use `Val(:kernel)`, `Val(:blas)`, and
`Val(:triangularsolve)`. Extensions should add methods only for backend and operand
combinations they implement; the generic methods provide the fallback behavior.

# Built-in backends

  - `Val(:kernel)`: runs `:lower`/`:upper` through the in-tree column-oriented
    kernels; `:factor_right_upper`/`:factor_lower` are forwarded to
    `Val(:triangularsolve)`.
  - `Val(:blas)`: runs `:lower`/`:upper` through `BLAS.trsm!` when `W` and `B` are
    strided `BlasFloat` matrices, and through the kernels otherwise;
    `:factor_right_upper`/`:factor_lower` are forwarded to `Val(:triangularsolve)`.
  - `Val(:triangularsolve)`: the in-tree method runs `:factor_right_upper` through
    `LinearAlgebra.rdiv!` with `UpperTriangular` and `:factor_lower` through
    `LinearAlgebra.ldiv!` with `UnitLowerTriangular`, and forwards
    `:lower`/`:upper` to `Val(:blas)`.

This is the extension hook: a package can add a more specific method for
`Val(:triangularsolve)` and supported panel types. The
`LinearSolveRecursiveFactorizationExt` extension defines

    supernodal_panel_solve_backend!(::Val{:triangularsolve}, W::StridedMatrix{Tv},
        B::StridedMatrix{Tv}, np::Int; operation::Symbol) where {Tv <: Union{Float32, Float64}}

which runs all four operations through TriangularSolve.jl and is active once
RecursiveFactorization.jl and TriangularSolve.jl are both loaded
(`using RecursiveFactorization` is enough, since it depends on TriangularSolve.jl).
"""
function supernodal_panel_solve_backend! end
# after default.jl: the vendored solver caches its dense diagonal blocks
# with LinearSolve's own default solver, so it needs DefaultLinearSolver{,Init}
include("SupernodalLU/SupernodalLU.jl")
include("init.jl")
include("adjoint.jl") # LinearSolveAdjoint struct definition only; rrules are in ChainRulesCore ext

## Deprecated, remove in July 2025

@static if isdefined(SciMLBase, :DiffEqArrayOperator)
    function defaultalg(
            A::SciMLBase.DiffEqArrayOperator, b,
            assump::OperatorAssumptions{Bool}
        )
        defaultalg(A.A, b, assump)
    end
end

@inline function _notsuccessful(
        F::LinearAlgebra.QRCompactWY{
            T, A,
        }
    ) where {T, A <: GPUArraysCore.AnyGPUArray}
    return hasmethod(LinearAlgebra.issuccess, (typeof(F),)) ?
        !LinearAlgebra.issuccess(F) : false
end

@inline function _notsuccessful(F::LinearAlgebra.QRCompactWY)
    (m, n) = size(F)
    U = view(F.factors, 1:min(m, n), 1:n)
    return any(iszero, Iterators.reverse(@view U[diagind(U)]))
end
@inline _notsuccessful(F) = hasmethod(LinearAlgebra.issuccess, (typeof(F),)) ?
    !LinearAlgebra.issuccess(F) : false

"""
    _qr_rank_deficient(F)

Cheap `O(min(m, n))` rank-deficiency test for an *unpivoted* QR factorization,
using the same relative threshold LAPACK's `xGELSY` (and therefore `A \\ b`) uses
to truncate the rank: a factorization is called rank-deficient when the smallest
`|R[i, i]|` falls at or below `min(m, n) * eps * max|R[i, i]|`.

This scans the `R` diagonal rather than reading a LAPACK `info` because
unpivoted QR has none to read: `geqrf`/`geqrt` only set `info < 0` for an
illegal argument and return `info == 0` on an exactly rank-deficient matrix, and
`QRCompactWY` correspondingly stores only `factors` and `T` with no `issuccess`
method. That is the same reason `_notsuccessful(::QRCompactWY)` above hand-scans
for an exact zero on the diagonal; this is that scan with a relative threshold,
so it also catches a merely negligible entry.

Unpivoted QR does not order the diagonal of `R` by magnitude, so this is a
heuristic rather than a rank-revealing test: it can miss a deficiency that only
column pivoting would expose (Kahan-style matrices). It exists so the default
algorithm can keep unpivoted QR on the fast path and only pay for a pivoted
refactorization when the cheap check says the answer would otherwise be garbage;
see `_default_qr_solve_with_fallback`. Factorization types the test does not
apply to (SPQR, GPU) return `false` and keep their existing behavior.

Being a threshold test, it is decisive only away from the threshold. For a matrix
sitting *on* the cutoff -- say a column scaled to roughly `1e-14` of another, where
the corresponding `R` diagonal entry lands at eps-level noise -- whether this fires
depends on rounding, and the answer can differ from `A \\ b`, which reaches the
same cutoff through a genuinely rank-revealing pivoted QR with better numerics.
Clearly rank-deficient input is handled; input engineered to sit at the boundary is
inherently ambiguous, and callers who need a decision there should ask for
`QRFactorization(ColumnNorm())` or `SVDFactorization()` directly.
"""
@inline _qr_rank_deficient(F) = false

# GPU factorizations cannot be indexed elementwise. Mirrors the GPU method on
# `_notsuccessful` above, and is constrained on the same `T` as the scanning
# method below so that it is strictly more specific (no dispatch ambiguity).
@inline function _qr_rank_deficient(
        ::LinearAlgebra.QRCompactWY{
            T, A,
        }
    ) where {T <: BLASELTYPES, A <: GPUArraysCore.AnyGPUArray}
    return false
end

@inline function _qr_rank_deficient(
        F::LinearAlgebra.QRCompactWY{
            T, A,
        }
    ) where {T <: BLASELTYPES, A}
    (m, n) = size(F)
    mn = min(m, n)
    mn == 0 && return false
    R = view(F.factors, 1:mn, 1:n)
    dmin = typemax(real(T))
    dmax = zero(real(T))
    for x in @view R[diagind(R)]
        a = abs(x)
        a < dmin && (dmin = a)
        a > dmax && (dmax = a)
    end
    return dmin <= mn * eps(real(T)) * dmax
end

# Solver Specific Traits
## Needs Square Matrix
"""
    needs_square_A(alg)

Returns `true` if the algorithm requires a square matrix.

`init` enforces this: an algorithm that returns `true` and is handed a
non-square `A` throws an `ArgumentError` naming the algorithms that do solve
least-squares/minimum-norm systems, rather than letting a `DimensionMismatch`
(or worse) escape from inside the factorization. Anything not listed below falls
back to the conservative `true`, so a new algorithm is rejected on non-square
input until it is declared otherwise.
"""
needs_square_A(::Nothing) = false  # Linear Solve automatically will use a correct alg!
needs_square_A(alg::SciMLLinearSolveAlgorithm) = true
for alg in (
        # Same reason as the `Nothing` method above: by the time `init` runs, an
        # unspecified algorithm has already been resolved to a
        # `DefaultLinearSolver`, and the polyalgorithm picks a non-square-capable
        # algorithm (pivoted QR, sparse column-pivoted QR, or a least-squares
        # Krylov method) for a non-square `A` itself.
        :DefaultLinearSolver,
        :QRFactorization, :FastQRFactorization, :NormalCholeskyFactorization,
        :NormalBunchKaufmanFactorization,
        # Rank-revealing/least-squares capable: `svd` and the column-pivoted
        # sparse QR both accept a non-square `A` and return the same answer as
        # `A \ b`. `SparseColumnPivotedQRFactorization` is in fact the default
        # for non-square sparse systems, so it must not be rejected here.
        :SVDFactorization, :SparseColumnPivotedQRFactorization,
        # Rank-revealing column-pivoted QR (`geqp3`): documented to return the
        # least-squares solution for any shape, including rank-deficient.
        :SpecializedQRFactorization,
        # QR-based GPU offloads. These cannot be exercised on a machine without
        # the corresponding GPU, and a QR is least-squares capable in principle,
        # so they are declared permissive: enforcing `true` here would newly
        # reject a shape that may work today.
        :CudaOffloadQRFactorization, :AMDGPUOffloadQRFactorization,
        :CudaOffloadFactorization,
        # A wrapper around a user-supplied factorization: whether a non-square
        # `A` is allowed depends on `fact_alg`, and the default (`factorize`) as
        # well as the obvious choices (`qr`, `svd`) all handle it. Leave the
        # rejection to the wrapped factorization.
        :GenericFactorization,
    )
    @eval needs_square_A(::$(alg)) = false
end
for kralg in (
        Krylov.lsmr!, Krylov.craigmr!, Krylov.lsqr!, Krylov.cgls!,
        Krylov.crls!, Krylov.cgne!, Krylov.craig!, Krylov.lslq!,
        Krylov.crmr!, Krylov.lnlq!,
    )
    @eval needs_square_A(::KrylovJL{$(typeof(kralg))}) = false
end
for alg in (
        :LUFactorization, :FastLUFactorization,
        :GenericLUFactorization, :SimpleLUFactorization,
        :RFLUFactorization, :ButterflyFactorization, :UMFPACKFactorization, :KLUFactorization, :SparspakFactorization,
        :DiagonalFactorization, :CholeskyFactorization, :BunchKaufmanFactorization,
        :CHOLMODFactorization, :LDLtFactorization, :AppleAccelerateLUFactorization,
        :MKLLUFactorization, :MetalLUFactorization, :CUSOLVERRFFactorization, :ParUFactorization,
        :HSLMA57Factorization, :HSLMA97Factorization,
        :STRUMPACKFactorization,
    )
    @eval needs_square_A(::$(alg)) = true
end

const IS_OPENBLAS = Ref(true)
isopenblas() = IS_OPENBLAS[]

const HAS_APPLE_ACCELERATE = Ref(false)
appleaccelerate_isavailable() = HAS_APPLE_ACCELERATE[]

# Extension availability checking functions
# Argument is simply to allow for a new dispatch to be added
useblis(x) = false
usecuda(x) = false
usemetal(x) = false

# Formerly ext/LinearSolveSparseArraysExt.jl; kept as one excisable unit, see its
# header. Order matters both ways: after the traits above, which its own workload
# calls, and before the workload below, which it would otherwise invalidate.
include("sparsearrays.jl")

PrecompileTools.@compile_workload begin
    A = rand(4, 4)
    b = rand(4)
    prob = LinearProblem(A, b)
    sol = solve(prob)
    sol = solve(prob, LUFactorization())
    sol = solve(prob, KrylovJL_GMRES())
    # 80 x 80 is past both `GenericLUFactorization` size switches: the blocked
    # driver (panel, trsm, row swaps) and the register-blocked Schur kernel,
    # neither of which the 4 x 4 problem above reaches.
    Ablocked = rand(80, 80) + 80I
    bblocked = rand(80)
    sol = solve(LinearProblem(Ablocked, bblocked), GenericLUFactorization())
end

ALREADY_WARNED_CUDSS = Ref{Bool}(false)
error_no_cudss_lu(A) = nothing
cudss_loaded(A) = false
is_cusparse(A) = false
is_cusparse_csr(A) = false

is_cusparse_csc(A) = false

export LUFactorization, SVDFactorization, QRFactorization, GenericFactorization,
    LowRankUpdatedMatrix,
    GenericLUFactorization, GESVFactorization, SimpleLUFactorization,
    RFLUFactorization, ButterflyFactorization,
    NormalCholeskyFactorization, NormalBunchKaufmanFactorization,
    UMFPACKFactorization, KLUFactorization, PureKLUFactorization,
    SupernodalLUFactorization, supernodal_panel_solve!, supernodal_panel_solve_backend!,
    PureUMFPACKFactorization, SparseColumnPivotedQRFactorization, FastLUFactorization,
    FastQRFactorization,
    SparspakFactorization, DiagonalFactorization, CholeskyFactorization,
    BunchKaufmanFactorization, CHOLMODFactorization, LDLtFactorization,
    CUSOLVERRFFactorization, CliqueTreesFactorization, ParUFactorization,
    AMGXPreconditioner,
    STRUMPACKFactorization, MUMPSFactorization, SuperLUDISTFactorization,
    SpecializedLUFactorization, SpecializedQRFactorization,
    HSLMA57Factorization, HSLMA97Factorization

export LHLFactorization, update_gamma!

export LinearSolveFunction, DirectLdiv!, show_algorithm_choices

export KrylovJL, KrylovJL_CG, KrylovJL_MINRES, KrylovJL_GMRES,
    KrylovJL_BICGSTAB, KrylovJL_LSMR, KrylovJL_CRAIGMR, KrylovJL_FGMRES, WarmStart,
    ConjugateGradientsJL, ConjugateGradientsJL_CG, ConjugateGradientsJL_BICGSTAB,
    IterativeSolversJL, IterativeSolversJL_CG, IterativeSolversJL_GMRES,
    IterativeSolversJL_BICGSTAB, IterativeSolversJL_MINRES, IterativeSolversJL_IDRS,
    KrylovKitJL, KrylovKitJL_CG, KrylovKitJL_GMRES, KrylovJL_MINARES, AlgebraicMultigridJL

export ElementalJL

export GinkgoJL, GinkgoJL_CG, GinkgoJL_GMRES

export SimpleGMRES

export HYPREAlgorithm
export PETScAlgorithm
export PartitionedSolversAlgorithm
export CudaOffloadFactorization
export CudaOffloadLUFactorization
export CudaOffloadQRFactorization
export CUDAOffload32MixedLUFactorization
export AMDGPUOffloadLUFactorization, AMDGPUOffloadQRFactorization
export MKLPardisoFactorize, MKLPardisoIterate
export PanuaPardisoFactorize, PanuaPardisoIterate
export PardisoJL
export MUMPSFactorization
export SuperLUDISTFactorization
export MKLLUFactorization
export OpenBLASLUFactorization
export BLISLUFactorization
export OpenBLAS32MixedLUFactorization
export MKL32MixedLUFactorization
export AppleAccelerateLUFactorization
export AppleAccelerate32MixedLUFactorization
export RF32MixedLUFactorization
export MetalLUFactorization
export MetalOffload32MixedLUFactorization

export OperatorAssumptions, OperatorCondition, NonstructuralZeros

export LinearSolveAdjoint

export LinearVerbosity

export AbstractEigenvalueAlgorithm,
    DenseEigen, ArpackJL, ArnoldiMethod, ArnoldiMethodJL,
    KrylovKitEigen, JacobiDavidsonJL

end
