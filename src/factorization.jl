@generated function SciMLBase.solve!(
        cache::LinearCache, alg::AbstractFactorization;
        kwargs...
    )
    return quote
        A = convert(AbstractMatrix, cache.A)
        check_safety = _get_residualsafety(alg) && cache.isfresh
        # Back up A before in-place LU when:
        #   - residualsafety is enabled (for residual check using original A), OR
        #   - the default solver has safetyfallback (for restoring A after LU failure)
        needs_backup = check_safety ||
            (cache.alg isa DefaultLinearSolver && cache.alg.safetyfallback && cache.isfresh)
        A_original = needs_backup ? _copy_A_for_safety(cache) : A

        if cache.isfresh
            fact = do_factorization(alg, cache.A, cache.b, cache.u)
            cache.cacheval = fact

            # If factorization was not successful, return failure. Don't reset `isfresh`
            if _notsuccessful(fact)
                @SciMLMessage(
                    "Solver failed", cache.verbose,
                    :solver_failure
                )
                return SciMLBase.build_linear_solution(
                    alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
                )
            end

            cache.isfresh = false
        end

        y = _ldiv!(
            cache.u, @get_cacheval(cache, $(Meta.quot(defaultalg_symbol(alg)))),
            cache.b
        )

        if check_safety
            failed = _check_residual_safety(cache, alg, A_original, y)
            failed !== nothing && return failed
        end

        return SciMLBase.build_linear_solution(
            alg, y, nothing, nothing; retcode = ReturnCode.Success
        )
    end
end

macro get_cacheval(cache, algsym)
    return quote
        if $(esc(cache)).alg isa DefaultLinearSolver
            getfield($(esc(cache)).cacheval, $algsym)
        else
            $(esc(cache)).cacheval
        end
    end
end

# Normalize deprecated Val-based pivot arguments to PivotingStrategy types.
# Julia 1.12 deprecated Val(true)/Val(false) in favor of RowMaximum()/NoPivot().
_normalize_pivot(pivot::LinearAlgebra.PivotingStrategy) = pivot
_normalize_pivot(::Val{true}) = RowMaximum()
_normalize_pivot(::Val{false}) = NoPivot()

const PREALLOCATED_IPIV = Vector{LinearAlgebra.BlasInt}(undef, 0)
const PREALLOCATED_RESIDUAL = Vector{Float64}(undef, 0)

# Trait for checking if an algorithm has residualsafety enabled
_get_residualsafety(alg) = false
# Methods for extension_algs.jl types (defined before this file is included)
_get_residualsafety(alg::RFLUFactorization) = alg.residualsafety
_get_residualsafety(alg::BLISLUFactorization) = alg.residualsafety
_get_residualsafety(alg::CudaOffloadLUFactorization) = alg.residualsafety
_get_residualsafety(alg::MetalLUFactorization) = alg.residualsafety

_typed_copy(A) = copy(A)
_typed_copy(A::Adjoint) = adjoint(copy(parent(A)))
_typed_copy(A::Transpose) = transpose(copy(parent(A)))

"""
    _copy_A_for_safety(cache::LinearCache)

Save a copy of `cache.A` before in-place LU factorization modifies it, for use in
post-solve residual checking and QR fallback restoration.

When inside `DefaultLinearSolver`, reuses `A_backup` in `DefaultLinearSolverInit`.
On the first call, `A_backup` aliases `cache.A` (for type stability at init), so a
separate buffer is allocated and stored. Subsequent calls reuse this buffer via
`copyto!` (non-allocating after warmup). For standalone use, allocates a copy.
"""
function _copy_A_for_safety(cache::LinearCache)
    if cache.alg isa DefaultLinearSolver
        cv = cache.cacheval
        A = cache.A
        if !cv.a_backup_allocated || size(cv.A_backup) != size(A)
            # First call or size mismatch: allocate a private buffer.
            # A_backup initially aliases prob.A so we must not copyto! into it.
            cv.A_backup = _typed_copy(A)
            cv.a_backup_allocated = true
        else
            # Reuse existing private buffer (non-allocating).
            copyto!(cv.A_backup, A)
        end
        cv.a_backup_synced = true
        return cv.A_backup
    else
        return _typed_copy(cache.A)
    end
end

"""
    _check_residual_safety(cache::LinearCache, alg, A_original, y; iters = 0, resid = nothing)

Post-solve residual check for LU algorithms with `residualsafety=true`.
Computes `‖A*y - b‖` and returns an `APosterioriSafetyFailure` solution if it
exceeds `abstol + reltol * ‖b‖`. Returns `nothing` if the residual is acceptable.

Iterative callers can pass their backend's convergence metadata through `iters`
and `resid` so a failing check still reports it. The defaults keep the failure
solution type-identical to the success path (`resid = nothing`): substituting the
check's own `res_norm` here would put a `Float64`/`Nothing` union in `solve!`'s
return type for every LU algorithm, breaking the concrete-return QA invariant.

When inside `DefaultLinearSolver`, uses the pre-allocated `residual_buf` from
`DefaultLinearSolverInit` (non-allocating). For standalone use, allocates a buffer.
"""
function _check_residual_safety(
        cache::LinearCache, alg, A_original, y; iters::Int = 0, resid = nothing
    )
    b = cache.b
    if cache.alg isa DefaultLinearSolver
        buf = cache.cacheval.residual_buf
        if size(buf) != size(b)
            # `resize!` only applies to vectors; matrix (batched) b just allocates.
            buf = buf isa Vector && b isa AbstractVector ? resize!(buf, length(b)) :
                similar(b)
        end
    else
        buf = similar(b)
    end
    mul!(buf, A_original, y)
    axpy!(-one(eltype(buf)), b, buf)
    res_norm = norm(buf)
    b_norm = norm(b)
    tol = cache.abstol + cache.reltol * b_norm
    if res_norm > tol
        @SciMLMessage(cache.verbose, :residual_safety) do
            return "Residual safety check failed: ‖A*x - b‖ = $(res_norm), tol = $(tol) (abstol = $(cache.abstol), reltol = $(cache.reltol), ‖b‖ = $(b_norm), ratio = $(res_norm / tol))"
        end
        return SciMLBase.build_linear_solution(
            alg, y, resid, nothing;
            retcode = ReturnCode.APosterioriSafetyFailure, iters
        )
    end
    return nothing
end

_ldiv!(x, A, b) = ldiv!(x, A, b)

_ldiv!(x, A, b::SVector) = (x .= A \ b)
_ldiv!(::SVector, A, b::SVector) = (A \ b)
_ldiv!(::SVector, A, b) = (A \ b)

function _direct_lu_factorize! end
function _direct_lu_solve! end

# Build a column-pivoted sparse QR factorization of `A` (the default sparse-LU
# singular fallback). The method is provided by `src/sparsearrays.jl` over
# SparseColumnPivotedQR.jl; this generic declaration lets `src/default.jl` call it.
function sparse_colpivqr_factorize end

# Heuristic shared by the sparse default's LU and QR choices: `true` selects the
# pure-Julia "KLU-style" solver for less-structured problems (small, or medium and
# very sparse) — `PureKLUFactorization` for LU and `SparseColumnPivotedQRFactorization`
# for QR — while `false` selects the SuiteSparse solver for more structure (UMFPACK
# for LU, SPQR for QR). `src/sparsearrays.jl` provides the real method for
# sparse matrices; the generic fallback prefers the pure-Julia option.
use_klulike_sparse_structure(A, b) = true

# RF Bad fallback: will fail if `A` is just a stand-in
# This should instead just create the factorization type.
function init_cacheval(
        alg::AbstractFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return do_factorization(alg, convert(AbstractMatrix, A), b, u)
end

## RFLU Factorization

function LinearSolve.init_cacheval(
        alg::RFLUFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    # `solve!` stores `(RecursiveFactorization.lu!(A, ipiv, ...), ipiv)` with this
    # `Vector{BlasInt}` pivot; rebuild the instance with it so the cacheval slot
    # type matches for dense CPU arrays whose container isn't `Base.Array`
    # (e.g. `FixedSizeArray`), whose `lu_instance` pivot would otherwise differ.
    luinst = ArrayInterface.lu_instance(convert(AbstractMatrix, A))
    # `lu_instance` may return a non-`LinearAlgebra.LU` (e.g. `StaticArrays.LU`
    # for a `SizedMatrix`, with fields `L`/`U`/`p` rather than `factors`/`info`);
    # those already carry a `Vector` pivot, so use them as-is.
    luinst isa LinearAlgebra.LU || return luinst, ipiv
    return LinearAlgebra.LU(luinst.factors, ipiv, luinst.info), ipiv
end

function LinearSolve.init_cacheval(
        alg::RFLUFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_LU, PREALLOCATED_IPIV
end

function LinearSolve.init_cacheval(
        alg::RFLUFactorization,
        A::Union{Diagonal, SymTridiagonal, Tridiagonal}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing, nothing
end

## LU Factorizations

"""
`LUFactorization(pivot=LinearAlgebra.RowMaximum())`

Julia's built in `lu`. Equivalent to calling `lu!(A)`

  - On dense matrices, this uses the current BLAS implementation of the user's computer,
    which by default is OpenBLAS but will use MKL if the user does `using MKL` in their
    system.
  - On sparse matrices, this will use UMFPACK from SparseArrays. Note that this will not
    cache the symbolic factorization.
  - On CuMatrix, it will use a CUDA-accelerated LU from CuSolver.
  - On BandedMatrix and BlockBandedMatrix, it will use a banded LU.

The back-solve is size-aware for LAPACK-eligible dense factors: single-vector solves at
or below a measured crossover (`N = 256` for `Matrix` factors) use the same pure-Julia
triangular sweeps as `GenericLUFactorization`, which beat `getrs!`'s call and blocking
overhead at those sizes; larger vectors and matrix right-hand sides use `ldiv!`
(LAPACK `getrs!`).

## Positional Arguments

  - pivot: The choice of pivoting. Defaults to `LinearAlgebra.RowMaximum()`. The other choice is
    `LinearAlgebra.NoPivot()`.
"""
Base.@kwdef struct LUFactorization{P} <: AbstractDenseFactorization
    pivot::P = LinearAlgebra.RowMaximum()
    reuse_symbolic::Bool = true
    check_pattern::Bool = true # Check factorization re-use
    residualsafety::Bool = false
end

# Legacy dispatch
LUFactorization(pivot) = LUFactorization(; pivot = RowMaximum())

"""
`GenericLUFactorization(pivot=LinearAlgebra.RowMaximum())`

Julia's built in generic LU factorization. Equivalent to calling LinearAlgebra.generic_lufact!.
Supports arbitrary number types. Has low overhead and is good for small matrices.

For `StridedMatrix{Float64}`/`StridedMatrix{Float32}` with `RowMaximum` pivoting the
factorization runs a blocked pure-Julia kernel (panel factorization + packed Schur
update, the LAPACK `getrf` structure) instead of the scalar textbook loop. The Schur
update is a register-blocked microkernel written on Base's `VecElement` tuples and
LLVM's `fmuladd` intrinsic, so the algorithm stays dependency-free — no
LoopVectorization, no SIMD.jl, no CPU feature detection — while scaling far beyond the
scalar kernel: on machines with a weak or unavailable BLAS it is competitive with (or
faster than) OpenBLAS/MKL `getrf` up to a few hundred unknowns. Other element types and
pivots keep the scalar generic path.

The back-solve is a pure-Julia column-oriented triangular solve (apply `ipiv`, unit-lower
forward, upper backward) rather than `ldiv!(::LU, ·)` / OpenBLAS `getrs!`. That avoids the
~290 ns BLAS call floor that otherwise dominates at the small-`N` sizes where this algorithm
is the default (see SciML/LinearSolve.jl#1145). `Adjoint`/`Transpose` operators, whose
factors are stored through the lazy wrapper, get an orientation-specialized kernel that
traverses the underlying parent column-contiguously (see SciML/LinearSolve.jl#1159).

The generic back-solve is used at every size, with no BLAS deferral: explicitly selecting
this algorithm selects the pure-Julia path for both the factorization and the solve. Note
that above `N ≈ 400` a single-vector generic sweep is slower than `getrs!`; if BLAS is
acceptable at large sizes, use `LUFactorization` (whose back-solve is size-aware) or the
default algorithm instead.

## Positional Arguments

  - pivot: The choice of pivoting. Defaults to `LinearAlgebra.RowMaximum()`. The other choice is
    `LinearAlgebra.NoPivot()`.
"""
struct GenericLUFactorization{P} <: AbstractDenseFactorization
    pivot::P
    residualsafety::Bool
end

GenericLUFactorization(pivot = RowMaximum(); residualsafety::Bool = false) = GenericLUFactorization(pivot, residualsafety)

mutable struct _GenericLUFactorizationCache{F, I, W}
    fact::F
    ipiv::I
    workspace::W
end

_generic_lu_workspace(A, pivot) = nothing

function _generic_lu_workspace(A::StridedMatrix{T}, ::RowMaximum) where {T <: Union{Float32, Float64}}
    m, n = size(A)
    minmn = min(m, n)
    nb = _blocked_lu_default_panel(minmn)
    return Vector{T}(undef, _blocked_lu_pack_size(m, n, minmn, nb))
end

function _generic_lufact!(A, pivot, ipiv, ::Nothing; kwargs...)
    return generic_lufact!(A, pivot, ipiv; kwargs...)
end

function _generic_lufact!(A, pivot, ipiv, workspace::Vector; kwargs...)
    return generic_lufact!(A, pivot, ipiv, workspace; kwargs...)
end

function _generic_lu_solve!(cacheval, A, u, b, pivot, isfresh::Bool)
    if isfresh
        fact = _generic_lufact!(
            A, pivot, cacheval.ipiv, cacheval.workspace; check = false
        )
        cacheval.fact = fact
        LinearAlgebra.issuccess(fact) || return false
    end
    fact = cacheval.fact
    if fact isa LinearAlgebra.LU
        _generic_lu_ldiv!(u, fact, b)
    else
        ldiv!(u, fact, b)
    end
    return true
end

_resize_generic_lu_workspace!(::Nothing, m::Int, n::Int) = nothing

function _resize_generic_lu_workspace!(workspace::Vector, m::Int, n::Int)
    minmn = min(m, n)
    nb = _blocked_lu_default_panel(minmn)
    resize!(workspace, _blocked_lu_pack_size(m, n, minmn, nb))
    return nothing
end

function _resize_generic_lu_cache!(cacheval::_GenericLUFactorizationCache, m::Int, n::Int)
    resize!(cacheval.ipiv, min(m, n))
    _resize_generic_lu_workspace!(cacheval.workspace, m, n)
    return nothing
end

resize_cacheval!(cache, cacheval::_GenericLUFactorizationCache, i) =
    _resize_generic_lu_cache!(cacheval, i, i)

function update_cacheval!(cache, cacheval::_GenericLUFactorizationCache, name::Symbol, A)
    name === :A && _resize_generic_lu_cache!(cacheval, size(A, 1), size(A, 2))
    return cacheval
end

# Pure-Julia LU back-solve used by GenericLUFactorization. A pivoted LU vector
# solve at small N is a handful of flops but ~290 ns through OpenBLAS/MKL
# getrs! (call overhead); the same algorithm written out as scalar loops is
# several times faster up through at least N≈80 (#1145). Always used for
# GenericLU — no BLAS cutoff — so the path stays dependency- and vendor-free.
# Multi-RHS applies the row permutation once, then runs the triangular sweeps
# per column (nrhs == 1 is the hot path for Newton/ODE).
@inline function _naive_lu_ldiv!(
        fac::AbstractMatrix, ipiv::AbstractVector, b::AbstractVector
    )
    n = length(b)
    @inbounds for i in eachindex(ipiv)
        p = ipiv[i]
        if p != i
            b[i], b[p] = b[p], b[i]
        end
    end
    @inbounds for j in 1:n
        bj = b[j]
        for i in (j + 1):n
            b[i] = muladd(-fac[i, j], bj, b[i])
        end
    end
    @inbounds for j in n:-1:1
        b[j] /= fac[j, j]
        bj = b[j]
        for i in 1:(j - 1)
            b[i] = muladd(-fac[i, j], bj, b[i])
        end
    end
    return b
end

@inline function _naive_lu_ldiv!(
        fac::AbstractMatrix, ipiv::AbstractVector, B::AbstractMatrix
    )
    n = size(B, 1)
    nrhs = size(B, 2)
    @inbounds for i in eachindex(ipiv)
        p = ipiv[i]
        if p != i
            for col in 1:nrhs
                B[i, col], B[p, col] = B[p, col], B[i, col]
            end
        end
    end
    @inbounds for j in 1:n
        for col in 1:nrhs
            bj = B[j, col]
            for i in (j + 1):n
                B[i, col] = muladd(-fac[i, j], bj, B[i, col])
            end
        end
    end
    @inbounds for j in n:-1:1
        invd = inv(fac[j, j])
        for col in 1:nrhs
            B[j, col] *= invd
            bj = B[j, col]
            for i in 1:(j - 1)
                B[i, col] = muladd(-fac[i, j], bj, B[i, col])
            end
        end
    end
    return B
end

# Orientation-specialized back-solve for the Adjoint/Transpose-wrapped strided
# factors that `init_cacheval` produces for adjoint/transpose operators (#1159).
# The column-oriented sweeps above would walk the wrapper's parent at stride N;
# these run the same solves in inner-product form so the parent is traversed
# column-contiguously. Indexing through the wrapper keeps conj correct for
# complex Adjoint factors.
const _AdjTransStridedFactors = Union{
    Adjoint{<:Any, <:StridedMatrix}, Transpose{<:Any, <:StridedMatrix},
}

@inline function _naive_lu_ldiv!(
        fac::_AdjTransStridedFactors, ipiv::AbstractVector, b::AbstractVector
    )
    n = length(b)
    @inbounds for i in eachindex(ipiv)
        p = ipiv[i]
        if p != i
            b[i], b[p] = b[p], b[i]
        end
    end
    @inbounds for i in 2:n
        acc = b[i]
        @simd for j in 1:(i - 1)
            acc = muladd(-fac[i, j], b[j], acc)
        end
        b[i] = acc
    end
    @inbounds for i in n:-1:1
        acc = b[i]
        @simd for j in (i + 1):n
            acc = muladd(-fac[i, j], b[j], acc)
        end
        b[i] = acc / fac[i, i]
    end
    return b
end

@inline function _naive_lu_ldiv!(
        fac::_AdjTransStridedFactors, ipiv::AbstractVector, B::AbstractMatrix
    )
    n = size(B, 1)
    nrhs = size(B, 2)
    @inbounds for i in eachindex(ipiv)
        p = ipiv[i]
        if p != i
            for col in 1:nrhs
                B[i, col], B[p, col] = B[p, col], B[i, col]
            end
        end
    end
    @inbounds for i in 2:n
        for col in 1:nrhs
            acc = B[i, col]
            @simd for j in 1:(i - 1)
                acc = muladd(-fac[i, j], B[j, col], acc)
            end
            B[i, col] = acc
        end
    end
    @inbounds for i in n:-1:1
        invd = inv(fac[i, i])
        for col in 1:nrhs
            acc = B[i, col]
            @simd for j in (i + 1):n
                acc = muladd(-fac[i, j], B[j, col], acc)
            end
            B[i, col] = acc * invd
        end
    end
    return B
end

# 3-arg form matching `ldiv!(x, F, b)`: copy when `x !== b`, then in-place solve.
# No size cutoff here, ever: choosing `GenericLUFactorization` is choosing the
# generic back-solve; the size-aware path is `_smart_lu_ldiv!` below.
function _generic_lu_ldiv!(x, F::LU, b)
    if x !== b
        copyto!(x, b)
    end
    return _naive_lu_ldiv!(F.factors, F.ipiv, x)
end

# Measured single-vector crossover vs `ldiv!` (min-times, 1 BLAS thread, EPYC
# 7502): naive wins to ≈ N 400 on `Matrix` factors and ≈ 900 on the wrapped
# orientations, then loses 1.14x/1.24x/1.35x at N = 512/1000/2000 (`Matrix`).
_naive_ldiv_cutoff(::AbstractMatrix) = 256
_naive_ldiv_cutoff(::_AdjTransStridedFactors) = 512

# Branch predicate of `_smart_lu_ldiv!` below. Split out so the selection is
# observable directly: the naive kernel and `getrs!` agree to within an ulp, so
# comparing their outputs bitwise is not a reliable way to tell which one ran.
_use_naive_lu_ldiv(x, F, b) = false

function _use_naive_lu_ldiv(
        x::StridedVector{T},
        F::LU{T, <:Union{StridedMatrix{T}, _AdjTransStridedFactors}},
        b::StridedVector{T}
    ) where {T <: BLASELTYPES}
    return !(
        x isa GPUArraysCore.AnyGPUArray || b isa GPUArraysCore.AnyGPUArray ||
            F.factors isa GPUArraysCore.AnyGPUArray
    ) && length(x) <= _naive_ldiv_cutoff(F.factors)
end

# Size-aware back-solve for `ldiv!`-baseline algorithms (`LUFactorization` and
# the defaults routed to it): single vectors at or below the crossover take the
# naive kernel; everything else (larger vectors, multi-RHS, non-BLAS/GPU) keeps
# `ldiv!`. Explicit generic selections never consult this cutoff.
_smart_lu_ldiv!(x, F, b) = _ldiv!(x, F, b)

function _smart_lu_ldiv!(
        x::StridedVector{T},
        F::LU{T, <:Union{StridedMatrix{T}, _AdjTransStridedFactors}},
        b::StridedVector{T}
    ) where {T <: BLASELTYPES}
    _use_naive_lu_ldiv(x, F, b) || return _ldiv!(x, F, b)
    x !== b && copyto!(x, b)
    return _naive_lu_ldiv!(F.factors, F.ipiv, x)
end

# Trait methods for types defined in this file (must come after struct definitions)
_get_residualsafety(alg::LUFactorization) = alg.residualsafety
_get_residualsafety(alg::GenericLUFactorization) = alg.residualsafety

# Dense-LU refactorization buffer reuse: `lu`/`lu!` allocate a fresh pivot
# vector (and, without aliasing, a fresh factors copy) on every `cache.A = X`
# refactorization. When the cached `LU`'s buffers match the incoming matrix,
# factorize through `LAPACK.getrf!` (or, outside the LAPACK fast path, the
# vendored `generic_lufact!`) with the cached `ipiv` instead. The
# preallocated-`ipiv` `getrf!` method only exists on Julia >= 1.11; older
# releases keep the allocating LAPACK path, but `generic_lufact!` accepts a
# provided `ipiv` on every supported Julia version.
function _lu_cacheval_ipiv_matches(cacheval, A)
    return cacheval isa LU &&
        cacheval.ipiv isa Vector{BlasInt} &&
        length(cacheval.ipiv) == min(size(A)...)
end
@static if VERSION >= v"1.11"
    _reusable_lu_cacheval(cacheval, A) = _lu_cacheval_ipiv_matches(cacheval, A)
    function _lu_reusing_ipiv!(A, ipiv::Vector{BlasInt})
        factors, piv, info = LAPACK.getrf!(A, ipiv; check = false)
        return LU{eltype(factors), typeof(factors), typeof(piv)}(
            factors, piv, convert(BlasInt, info)
        )
    end
else
    _reusable_lu_cacheval(cacheval, A) = false
    _lu_reusing_ipiv!(A, ipiv) = error("unreachable: requires Julia >= 1.11")
end

function SciMLBase.solve!(cache::LinearCache, alg::LUFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    check_safety = alg.residualsafety && cache.isfresh
    needs_backup = check_safety ||
        (cache.alg isa DefaultLinearSolver && cache.alg.safetyfallback && cache.isfresh)
    A_original = needs_backup ? _copy_A_for_safety(cache) : A
    if cache.isfresh
        cacheval = @get_cacheval(cache, :LUFactorization)
        local fact
        try
            if issparsematrix(A) && alg.reuse_symbolic
                # Caches the symbolic factorization: https://github.com/JuliaLang/julia/pull/33738
                # If SparseMatrixCSC, check if the pattern has changed
                if alg.check_pattern && pattern_changed(cacheval, A)
                    fact = lu(A, check = false)
                else
                    fact = lu!(cacheval, A, check = false)
                end
            elseif cache.alias_A && !issparsematrix(A) &&
                    !(A isa GPUArraysCore.AnyGPUArray) &&
                    ArrayInterface.can_setindex(typeof(A))
                # The user permitted overwriting A (`alias_A = true` at `init`),
                # so refactorize in place and skip the O(n²) copy `lu` makes.
                pivot = _normalize_pivot(alg.pivot)
                if A isa StridedMatrix{<:LinearAlgebra.BlasFloat} &&
                        pivot isa RowMaximum
                    if _reusable_lu_cacheval(cacheval, A)
                        fact = _lu_reusing_ipiv!(A, cacheval.ipiv)
                    else
                        fact = lu!(A, pivot; check = false)
                    end
                elseif A isa StridedMatrix &&
                        pivot isa Union{RowMaximum, NoPivot, RowNonZero} &&
                        _lu_cacheval_ipiv_matches(cacheval, A)
                    # `lu!` on a strided matrix outside the LAPACK fast path runs
                    # `generic_lufact!`, which would allocate a fresh pivot
                    # vector; call it directly with the cached one instead.
                    fact = generic_lufact!(A, pivot, cacheval.ipiv; check = false)
                else
                    fact = lu!(A, pivot; check = false)
                end
            else
                pivot = _normalize_pivot(alg.pivot)
                if A isa StridedMatrix{<:LinearAlgebra.BlasFloat} &&
                        pivot isa RowMaximum && _reusable_lu_cacheval(cacheval, A) &&
                        cacheval.factors isa Matrix{eltype(A)} &&
                        size(cacheval.factors) == size(A) && cacheval.factors !== A
                    # A must stay intact, but the previous factorization's
                    # buffers can be overwritten in place of `lu`'s fresh copy.
                    copyto!(cacheval.factors, A)
                    fact = _lu_reusing_ipiv!(cacheval.factors, cacheval.ipiv)
                else
                    fact = lu(A, check = false)
                end
            end
        catch e
            # Some matrix types (e.g. BandedMatrix) throw LAPACKException on singular
            # matrices even with check=false, because their LAPACK wrappers don't
            # respect the check flag. Catch these and return Failure.
            if e isa LinearAlgebra.LAPACKException ||
                    e isa LinearAlgebra.SingularException
                @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
                return SciMLBase.build_linear_solution(
                    alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
                )
            else
                rethrow(e)
            end
        end
        cache.cacheval = fact

        if hasmethod(LinearAlgebra.issuccess, Tuple{typeof(fact)}) &&
                !LinearAlgebra.issuccess(fact)
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end

        cache.isfresh = false
    end

    F = @get_cacheval(cache, :LUFactorization)
    y = _smart_lu_ldiv!(cache.u, F, cache.b)

    if check_safety
        failed = _check_residual_safety(cache, alg, A_original, y)
        failed !== nothing && return failed
    end

    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

function do_factorization(alg::LUFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    pivot = _normalize_pivot(alg.pivot)
    if issparsematrixcsc(A)
        fact = handle_sparsematrixcsc_lu(A)
    elseif A isa GPUArraysCore.AnyGPUArray
        fact = lu(A; check = false)
    elseif !ArrayInterface.can_setindex(typeof(A))
        fact = lu(A, pivot; check = false)
    else
        fact = lu!(A, pivot; check = false)
    end
    return fact
end

function init_cacheval(
        alg::GenericLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    A = convert(AbstractMatrix, A)
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    # `lu_instance` may type its pivot after `A`'s container, so rebuild stdlib
    # `LU` instances with the cache-owned `Vector{BlasInt}` pivot.
    luinst = ArrayInterface.lu_instance(A)
    # `lu_instance` may return a non-`LinearAlgebra.LU` (e.g. `StaticArrays.LU`
    # for a `SizedMatrix`, with fields `L`/`U`/`p` rather than `factors`/`info`);
    # those already carry a `Vector` pivot, so use them as-is.
    workspace = _generic_lu_workspace(A, alg.pivot)
    luinst isa LinearAlgebra.LU ||
        return _GenericLUFactorizationCache(luinst, ipiv, workspace)
    fact = LinearAlgebra.LU(luinst.factors, ipiv, luinst.info)
    return _GenericLUFactorizationCache(fact, ipiv, workspace)
end

function init_cacheval(
        alg::GenericLUFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    workspace = _generic_lu_workspace(A, alg.pivot)
    return _GenericLUFactorizationCache(PREALLOCATED_LU, ipiv, workspace)
end

function SciMLBase.solve!(
        cache::LinearSolve.LinearCache, alg::GenericLUFactorization;
        kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    check_safety = alg.residualsafety && cache.isfresh
    needs_backup = check_safety ||
        (cache.alg isa DefaultLinearSolver && cache.alg.safetyfallback && cache.isfresh)
    A_original = needs_backup ? _copy_A_for_safety(cache) : A
    cacheval = LinearSolve.@get_cacheval(cache, :GenericLUFactorization)

    if !_generic_lu_solve!(cacheval, A, cache.u, cache.b, alg.pivot, cache.isfresh)
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
        )
    end
    cache.isfresh = false
    y = cache.u

    if check_safety
        failed = _check_residual_safety(cache, alg, A_original, y)
        failed !== nothing && return failed
    end

    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

function init_cacheval(
        alg::LUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(convert(AbstractMatrix, A))
end

function init_cacheval(
        alg::LUFactorization,
        A::Union{<:Adjoint, <:Transpose}, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    error_no_cudss_lu(A)
    return lu(A; check = false)
end

function init_cacheval(
        alg::GenericLUFactorization,
        A::Union{<:Adjoint, <:Transpose}, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    error_no_cudss_lu(A)
    A isa GPUArraysCore.AnyGPUArray && return nothing
    ipiv = Vector{LinearAlgebra.BlasInt}(undef, min(size(A)...))
    fact = LinearAlgebra.generic_lufact!(_typed_copy(A), alg.pivot; check = false)
    return _GenericLUFactorizationCache(fact, ipiv, nothing)
end

const PREALLOCATED_LU = ArrayInterface.lu_instance(rand(1, 1))

function init_cacheval(
        alg::LUFactorization,
        A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return PREALLOCATED_LU
end

function init_cacheval(
        alg::LUFactorization,
        A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        alg::GenericLUFactorization,
        A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

## GESVFactorization

"""
`GESVFactorization()`

A dense LU factorize-and-solve in the style of LAPACK's `gesv` driver, tuned for
repeated solves. On a fresh matrix it factorizes with `lu!(A; check = false)`
(so a singular factor is reported through the return code, never thrown) and
solves into `u`; the factorization is cached, so subsequent solves that only
change `b` are an allocation-free triangular solve. Batched (matrix) right-hand
sides are handled natively.

With `alias_A = true` the factorization overwrites `cache.A` directly; otherwise
a workspace-owned buffer is refilled with `copyto!` on each refactorization and
the user's matrix is left untouched.

Only dense strided BLAS floating-point matrices
(`StridedMatrix{<:Union{Float32, Float64, ComplexF32, ComplexF64}}`) are
supported through the caching interface; other operator types throw an
informative error at `init`. Square StaticArrays problems have a dedicated
direct dispatch with the same semantics (one-shot solve, singular input
reported as `ReturnCode.Failure`, no factorization retained) — see the
StaticArrays tutorial page.
"""
struct GESVFactorization <: AbstractDenseFactorization end

function init_cacheval(
        alg::GESVFactorization, A::StridedMatrix{T}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T <: LinearAlgebra.BlasFloat}
    return LinearAlgebra.LU(Matrix{T}(undef, 0, 0), BlasInt[], zero(BlasInt))
end

function init_cacheval(
        alg::GESVFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    throw(
        ArgumentError(
            "GESVFactorization only supports dense strided BLAS floating-point \
            matrices (StridedMatrix{<:Union{Float32, Float64, ComplexF32, \
            ComplexF64}}); got $(typeof(A)). Use LUFactorization or the default \
            algorithm instead."
        )
    )
end

function SciMLBase.solve!(cache::LinearCache, alg::GESVFactorization; kwargs...)
    A = convert(AbstractMatrix, cache.A)
    luf = @get_cacheval(cache, :GESVFactorization)
    if cache.isfresh
        if cache.alias_A && A isa Matrix
            # The user permitted overwriting A, so factorize it in place.
            Atarget = A
        else
            Fbuf = getfield(luf, :factors)
            if size(Fbuf) != size(A)
                Fbuf = similar(A)
            end
            copyto!(Fbuf, A)
            Atarget = Fbuf
        end
        # `check = false` returns a singular factorization instead of throwing,
        # so a singular denominator is reported through the return code.
        luf = lu!(Atarget; check = false)
        cache.cacheval = luf
        if !LinearAlgebra.issuccess(luf)
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end
        cache.isfresh = false
    end
    # Solve into `u`; a matrix `b` (batched RHS) is handled natively by ldiv!.
    copyto!(cache.u, cache.b)
    ldiv!(luf, cache.u)
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing; retcode = ReturnCode.Success
    )
end

## QRFactorization

"""
`QRFactorization(pivot=LinearAlgebra.NoPivot(),blocksize=16)`

Julia's built in `qr`. Equivalent to calling `qr!(A)`.

  - On dense matrices, this uses the current BLAS implementation of the user's computer
    which by default is OpenBLAS but will use MKL if the user does `using MKL` in their
    system.
  - On sparse matrices, this will use SPQR from SparseArrays
  - On CuMatrix, it will use a CUDA-accelerated QR from CuSolver.
  - On BandedMatrix and BlockBandedMatrix, it will use a banded QR.

With the default `NoPivot()` this is not rank-revealing, so it cannot solve a
rank-deficient (least-squares) system: it reports `ReturnCode.Failure` when a
diagonal entry of `R` is exactly zero, and can return an overflowing solution
when one is merely negligible. Pass `ColumnNorm()` for a rank-revealing
factorization that truncates the rank the way `A \\ b` does. The default
algorithm handles this automatically — it starts with the cheaper unpivoted QR
and re-solves with `QRFactorization(ColumnNorm())` if `A` turns out to be
rank-deficient.
"""
struct QRFactorization{P} <: AbstractDenseFactorization
    pivot::P
    blocksize::Int
    inplace::Bool
end

QRFactorization(inplace = true) = QRFactorization(NoPivot(), 16, inplace)

function QRFactorization(pivot::LinearAlgebra.PivotingStrategy, inplace::Bool = true)
    return QRFactorization(pivot, 16, inplace)
end

function do_factorization(alg::QRFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if ArrayInterface.can_setindex(typeof(A))
        # Sparse CSC (SPQR) does not accept a pivoting strategy, and CUDA's
        # `qr` does not accept extra args either. Use the no-arg `qr(A)`
        # form in those cases. For other CPU matrices, always pass
        # `alg.pivot` so the return type is determined by the static
        # `QRFactorization{P}` parameter (otherwise this branch returns
        # `Union{QRCompactWY, QRPivoted}` depending on `alg.inplace`).
        if A isa GPUArraysCore.AnyGPUArray || is_cusparse(A) || issparsematrixcsc(A)
            fact = qr(A)
        elseif alg.inplace
            if A isa Symmetric
                fact = qr(A, alg.pivot)
            else
                fact = qr!(A, alg.pivot)
            end
        else
            fact = qr(A, alg.pivot)
        end
    else
        fact = qr(A, alg.pivot)
    end
    return fact
end

function init_cacheval(
        alg::QRFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(convert(AbstractMatrix, A), alg.pivot)
end

function init_cacheval(
        alg::QRFactorization, A::Symmetric{<:Number, <:Array}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return qr(convert(AbstractMatrix, A), alg.pivot)
end

const PREALLOCATED_QR_ColumnNorm = ArrayInterface.qr_instance(rand(1, 1), ColumnNorm())

function init_cacheval(
        alg::QRFactorization{ColumnNorm}, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_QR_ColumnNorm
end

function init_cacheval(
        alg::QRFactorization, A::Union{<:Adjoint, <:Transpose}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    A isa GPUArraysCore.AnyGPUArray && return qr(A)
    return qr(A, alg.pivot)
end

const PREALLOCATED_QR_NoPivot = ArrayInterface.qr_instance(rand(1, 1))

function init_cacheval(
        alg::QRFactorization{NoPivot}, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_QR_NoPivot
end

function init_cacheval(
        alg::QRFactorization, A::AbstractSciMLOperator, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

## CholeskyFactorization

"""
`CholeskyFactorization(; pivot = nothing, tol = 0.0, shift = 0.0, perm = nothing)`

Julia's built in `cholesky`. Equivalent to calling `cholesky!(A)`.

## Keyword Arguments

  - pivot: defaluts to NoPivot, can also be RowMaximum.
  - tol: the tol argument in CHOLMOD. Only used for sparse matrices.
  - shift: the shift argument in CHOLMOD. Only used for sparse matrices.
  - perm: the perm argument in CHOLMOD. Only used for sparse matrices.
"""
struct CholeskyFactorization{P, P2} <: AbstractDenseFactorization
    pivot::P
    tol::Int
    shift::Float64
    perm::P2
end

function CholeskyFactorization(; pivot = nothing, tol = 0.0, shift = 0.0, perm = nothing)
    pivot === nothing && (pivot = NoPivot())
    return CholeskyFactorization(pivot, 16, shift, perm)
end

function do_factorization(alg::CholeskyFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    pivot = _normalize_pivot(alg.pivot)
    if issparsematrixcsc(A)
        fact = cholesky(A; shift = alg.shift, check = false, perm = alg.perm)
    elseif A isa GPUArraysCore.AnyGPUArray
        fact = cholesky(A; check = false)
    elseif pivot === NoPivot()
        fact = cholesky!(A, pivot; check = false)
    else
        fact = cholesky!(A, pivot; tol = alg.tol, check = false)
    end
    return fact
end

function init_cacheval(
        alg::CholeskyFactorization, A::SMatrix{S1, S2}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {S1, S2}
    return cholesky(A)
end

function init_cacheval(
        alg::CholeskyFactorization, A::GPUArraysCore.AnyGPUArray, b, u, Pl,
        Pr, maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return cholesky(A; check = false)
end

function init_cacheval(
        alg::CholeskyFactorization, A::AbstractArray{<:BLASELTYPES}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return if LinearSolve.is_cusparse_csc(A)
        nothing
    elseif LinearSolve.is_cusparse_csr(A) && !LinearSolve.cudss_loaded(A)
        nothing
    else
        ArrayInterface.cholesky_instance(convert(AbstractMatrix, A), _normalize_pivot(alg.pivot))
    end
end

const PREALLOCATED_CHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1), NoPivot())

function init_cacheval(
        alg::CholeskyFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return PREALLOCATED_CHOLESKY
end

function init_cacheval(
        alg::CholeskyFactorization,
        A::Union{Diagonal, AbstractSciMLOperator, AbstractArray}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

## LDLtFactorization

"""
    LDLtFactorization(shift = 0.0, perm = nothing)

Julia's built-in LDLᵀ factorization. Dense inputs use `ldlt!`; sparse inputs
use `ldlt!` with the supplied `shift` and `perm` keyword values.

This method is intended for Hermitian or symmetric linear systems where an LDLᵀ
factorization is appropriate.
"""
struct LDLtFactorization{T} <: AbstractDenseFactorization
    shift::Float64
    perm::T
end

function LDLtFactorization(shift = 0.0, perm = nothing)
    return LDLtFactorization(shift, perm)
end

function do_factorization(alg::LDLtFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    if !issparsematrixcsc(A)
        fact = ldlt!(A)
    else
        fact = ldlt!(A, shift = alg.shift, perm = alg.perm)
    end
    return fact
end

function init_cacheval(
        alg::LDLtFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        alg::LDLtFactorization, A::SymTridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.ldlt_instance(convert(AbstractMatrix, A))
end

## SVDFactorization

"""
    SVDFactorization(full=false, alg=nothing)
Julia's built-in `svd`. Equivalent to `svd!(A)`.

- On dense matrices, this uses the current BLAS implementation.
- When `alg = nothing`, the backend default SVD algorithm is used
  (required for CUDA compatibility).
"""
struct SVDFactorization{A} <: AbstractDenseFactorization
    full::Bool
    alg::A
end

SVDFactorization() = SVDFactorization(false, nothing)

function do_factorization(alg::SVDFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    return if ArrayInterface.can_setindex(typeof(A))
        if alg.alg === nothing
            return svd!(A; full = alg.full)
        else
            return svd!(A; full = alg.full, alg = alg.alg)
        end
    else
        if alg.alg === nothing
            return svd(A; full = alg.full)
        else
            return svd(A; full = alg.full, alg = alg.alg)
        end
    end
end

function init_cacheval(
        alg::SVDFactorization, A::Union{Matrix, SMatrix}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_SVD = ArrayInterface.svd_instance(rand(1, 1))

function init_cacheval(
        alg::SVDFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return PREALLOCATED_SVD
end

function init_cacheval(
        alg::SVDFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

## BunchKaufmanFactorization

"""
`BunchKaufmanFactorization(; rook = false)`

Julia's built in `bunchkaufman`. Equivalent to calling `bunchkaufman(A)`.
Only for Symmetric matrices.

## Keyword Arguments

  - rook: whether to perform rook pivoting. Defaults to false.
"""
Base.@kwdef struct BunchKaufmanFactorization <: AbstractDenseFactorization
    rook::Bool = false
end

function do_factorization(alg::BunchKaufmanFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = bunchkaufman!(A, alg.rook; check = false)
    return fact
end

function init_cacheval(
        alg::BunchKaufmanFactorization, A::Symmetric{<:Number, <:Matrix}, b,
        u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.bunchkaufman_instance(convert(AbstractMatrix, A))
end

const PREALLOCATED_BUNCHKAUFMAN = ArrayInterface.bunchkaufman_instance(
    Symmetric(
        rand(
            1,
            1
        )
    )
)

function init_cacheval(
        alg::BunchKaufmanFactorization,
        A::Symmetric{Float64, Matrix{Float64}}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return PREALLOCATED_BUNCHKAUFMAN
end

function init_cacheval(
        alg::BunchKaufmanFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

## GenericFactorization

"""
`GenericFactorization(;fact_alg=LinearAlgebra.factorize)`: Constructs a linear solver from a generic
factorization algorithm `fact_alg` which complies with the Base.LinearAlgebra
factorization API. Quoting from Base:

      * If `A` is upper or lower triangular (or diagonal), no factorization of `A` is
        required. The system is then solved with either forward or backward substitution.
        For non-triangular square matrices, an LU factorization is used.
        For rectangular `A` the result is the minimum-norm least squares solution computed by a
        pivoted QR factorization of `A` and a rank estimate of `A` based on the R factor.
        When `A` is sparse, a similar polyalgorithm is used. For indefinite matrices, the `LDLt`
        factorization does not use pivoting during the numerical factorization and therefore the
        procedure can fail even for invertible matrices.

## Keyword Arguments

  - fact_alg: the factorization algorithm to use. Defaults to `LinearAlgebra.factorize`, but can be
    swapped to choices like `lu`, `qr`
"""
struct GenericFactorization{F} <: AbstractDenseFactorization
    fact_alg::F
end

GenericFactorization(; fact_alg = LinearAlgebra.factorize) = GenericFactorization(fact_alg)

function do_factorization(alg::GenericFactorization, A, b, u)
    A = convert(AbstractMatrix, A)
    fact = alg.fact_alg(A)
    return fact
end

function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(lu)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(lu)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(qr)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(qr!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.qr_instance(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end

function init_cacheval(
        alg::GenericFactorization{typeof(svd)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)},
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::Diagonal, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::Tridiagonal, b, u, Pl,
        Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.svd_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(svd)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

function init_cacheval(
        alg::GenericFactorization, A::Diagonal, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization, A, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return init_cacheval(
        alg, convert(AbstractMatrix, A), b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
end
function init_cacheval(
        alg::GenericFactorization, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return do_factorization(alg, A, b, u)
end

function init_cacheval(
        alg::Union{
            GenericFactorization{typeof(bunchkaufman!)},
            GenericFactorization{typeof(bunchkaufman)},
        },
        A::Union{Hermitian, Symmetric}, b, u, Pl, Pr, maxiters::Int, abstol,
        reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return BunchKaufman(A.data, Array(1:size(A, 1)), A.uplo, true, false, 0)
end

function init_cacheval(
        alg::Union{
            GenericFactorization{typeof(bunchkaufman!)},
            GenericFactorization{typeof(bunchkaufman)},
        },
        A::StridedMatrix{<:LinearAlgebra.BlasFloat}, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if eltype(A) <: Complex
        return bunchkaufman!(Hermitian(A))
    else
        return bunchkaufman!(Symmetric(A))
    end
end

# Fallback, tries to make nonsingular and just factorizes
# Try to never use it.

# Cholesky needs the posdef matrix, for GenericFactorization assume structure is needed
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    newA = copy(convert(AbstractMatrix, A))
    return do_factorization(alg, newA, b, u)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::AbstractMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    newA = copy(convert(AbstractMatrix, A))
    return do_factorization(alg, newA, b, u)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)},
        A::Diagonal, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky!)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)},
        A::Diagonal, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return Diagonal(inv.(A.diag))
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::Tridiagonal, b, u, Pl, Pr,
        maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return ArrayInterface.lu_instance(A)
end
function init_cacheval(
        alg::GenericFactorization{typeof(cholesky)}, A::SymTridiagonal{T, V}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    ) where {T, V}
    return LinearAlgebra.LDLt{T, SymTridiagonal{T, V}}(A)
end

# Ambiguity handling dispatch

################################## Factorizations which require solve! overloads

"""
`UMFPACKFactorization(; reuse_symbolic = true, check_pattern = true,
                       max_iterative_refinement_steps = nothing)`

A fast sparse multithreaded LU-factorization which specializes on sparsity
patterns with “more structure”.

!!! note

    By default, the SparseArrays.jl are implemented for efficiency by caching the
    symbolic factorization. If the sparsity pattern of `A` may change between solves, set `reuse_symbolic=false`.
    If the pattern is assumed or known to be constant, set `reuse_symbolic=true` to avoid
    unnecessary recomputation. To further reduce computational overhead, you can disable
    pattern checks entirely by setting `check_pattern = false`. Note that this may error
    if the sparsity pattern does change unexpectedly.

## Iterative refinement

`max_iterative_refinement_steps` sets the maximum number of steps of iterative
refinement UMFPACK performs on each solve, trading time for accuracy on
ill-conditioned systems.

SuiteSparse defaults this to `2`, but Julia's SparseArrays turns it off
(JuliaLang/julia#122), and `nothing` (the default here) keeps whatever
SparseArrays itself defaults to rather than pinning a value. Pass a nonnegative
integer to choose explicitly: `2` restores SuiteSparse's default, `0` disables
refinement.

```julia
solve(prob, UMFPACKFactorization(max_iterative_refinement_steps = 2))
```

Refinement needs the original matrix, so it only applies while the factorization
is used with the matrix it was computed from.
"""
Base.@kwdef struct UMFPACKFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true # Check factorization re-use
    max_iterative_refinement_steps::Union{Nothing, Int} = nothing

    function UMFPACKFactorization(
            reuse_symbolic::Bool, check_pattern::Bool,
            max_iterative_refinement_steps::Union{Nothing, Int}
        )
        if max_iterative_refinement_steps !== nothing &&
                max_iterative_refinement_steps < 0
            throw(
                ArgumentError(
                    "`max_iterative_refinement_steps` must be nonnegative, got " *
                        "$max_iterative_refinement_steps"
                )
            )
        end
        return new(reuse_symbolic, check_pattern, max_iterative_refinement_steps)
    end
end

function init_cacheval(
        alg::UMFPACKFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

"""
`KLUFactorization(;reuse_symbolic=true, check_pattern=true)`

A fast sparse LU-factorization which specializes on sparsity patterns with “less structure”.

!!! note

    By default, the SparseArrays.jl are implemented for efficiency by caching the
    symbolic factorization. If the sparsity pattern of `A` may change between solves, set `reuse_symbolic=false`.
    If the pattern is assumed or known to be constant, set `reuse_symbolic=true` to avoid
    unnecessary recomputation. To further reduce computational overhead, you can disable
    pattern checks entirely by setting `check_pattern = false`. Note that this may error
    if the sparsity pattern does change unexpectedly.
"""
Base.@kwdef struct KLUFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true
end

function init_cacheval(
        alg::KLUFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

"""
`PureKLUFactorization(; reuse_symbolic = true, check_pattern = true, use_fma = true, fully_preallocated = nothing, tol = 0.001)`

A pure-Julia port of SuiteSparse's KLU sparse LU solver, provided by
[PureKLU.jl](https://github.com/SciML/PureKLU.jl). It has no SuiteSparse binary
dependency and supports generic element types in addition to `Float64`/`ComplexF64`.
Loading PureKLU (`using PureKLU`) makes this the default sparse LU for "less
structured" sparse matrices, replacing the SuiteSparse-backed [`KLUFactorization`](@ref)
in the default polyalgorithm.

!!! note

    `PureKLUFactorization` is only available once the `PureKLU` package is loaded
    (`using PureKLU`). It mirrors [`KLUFactorization`](@ref): by default the symbolic
    factorization is cached. If the sparsity pattern of `A` may change between solves,
    set `reuse_symbolic = false`. To skip the pattern check entirely (which errors if the
    pattern unexpectedly changes), set `check_pattern = false`.

## Keyword Arguments

  - `reuse_symbolic`: reuse the cached symbolic factorization across solves. Defaults to `true`.
  - `check_pattern`: check whether the sparsity pattern changed before reusing the
    symbolic factorization. Defaults to `true`.
  - `use_fma`: use fused multiply-add in the numeric kernel (faster, up to one ULP
    different from SuiteSparse KLU). Set to `false` for bit-for-bit agreement with
    SuiteSparse `KLUFactorization`. Defaults to `true`.
  - `fully_preallocated`: PureKLU's `fully_preallocated` option. `nothing` (default) lets
    PureKLU choose automatically based on the maximum block size.
  - `tol`: Pivot on a column's diagonal instead of largest entry if it is at least `tol` times
    larger in magnitude. Set `tol = 1.0` for partial pivoting, and `tol = 0.0` to always use the
    diagonal. Only applies to the initial factorization; refactorizations reuse the existing
    pivot ordering. Defaults to `0.001`.
"""
Base.@kwdef struct PureKLUFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true
    use_fma::Bool = true
    fully_preallocated::Union{Bool, Nothing} = nothing
    tol::Float64 = 0.001
end

function init_cacheval(
        alg::PureKLUFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

"""
`PureUMFPACKFactorization(; reuse_symbolic = true, check_pattern = true, throwerror = true)`

A pure-Julia port of SuiteSparse's UMFPACK unsymmetric sparse LU solver, provided
by [PureUMFPACK.jl](https://github.com/SciML/PureUMFPACK.jl). It has no SuiteSparse
binary dependency and supports generic element types in addition to
`Float64`/`ComplexF64`. It is the pure-Julia analogue of the SuiteSparse-backed
[`UMFPACKFactorization`](@ref).

!!! note

    `PureUMFPACKFactorization` is only available once the `PureUMFPACK` package is
    loaded (`using PureUMFPACK`). Unlike SuiteSparse UMFPACK, PureUMFPACK has no
    in-place numeric-refactorization (`lu!`-style) API: each fresh factorization
    recomputes the ordering, symbolic analysis, and numerics together. The
    `reuse_symbolic` and `check_pattern` keywords are accepted for API parity with
    [`UMFPACKFactorization`](@ref) and control caching of the factorization object
    across solves, but no symbolic factorization is shared between numeric refactors.

## Keyword Arguments

  - `reuse_symbolic`: reuse the cached factorization across solves when the sparsity
    pattern is unchanged. Defaults to `true`.
  - `check_pattern`: check whether the sparsity pattern changed before reusing the
    cached factorization. Defaults to `true`.
  - `throwerror`: whether to throw an error if PureUMFPACK.jl is not loaded. Defaults
    to `true`.
"""
struct PureUMFPACKFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool
    check_pattern::Bool

    function PureUMFPACKFactorization(
            ; reuse_symbolic = true, check_pattern = true, throwerror = true
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolvePureUMFPACKExt)
        return if throwerror && ext === nothing
            error("PureUMFPACKFactorization requires that PureUMFPACK is loaded, i.e. `using PureUMFPACK`")
        else
            new(reuse_symbolic, check_pattern)
        end
    end
end

function init_cacheval(
        alg::PureUMFPACKFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

"""
`SupernodalLUFactorization(; reuse_symbolic = true, check_pattern = true, ordering = :amd,
                            matching = :auto, eps_pivot = 1e-8, threaded = false)`

A pure-Julia implementation of the supernodal left–right-looking sparse LU
method of O. Schenk and K. Gärtner (FGCS 20(3), 2004; ETNA 23, 2006),
vendored self-contained in `src/SupernodalLU`: supernodal BLAS-3
LU on the symmetric pattern of `A + Aᵀ`, pivoting restricted to supernode
diagonal blocks with static pivot perturbation compensated by iterative
refinement, and maximum-weight matching + scaling preprocessing for
unsymmetric systems. No binary dependencies.

This is the strongest choice for "more structured" sparse systems (2D/3D
PDE-mesh-like patterns), where it outperforms both `UMFPACKFactorization`
and `KLUFactorization`. For very sparse circuit-like systems, prefer
`PureKLUFactorization`/`KLUFactorization`. Solves are allocation-free and
numeric refactorization on an unchanged sparsity pattern allocates only a
fixed ~64 B per cached supernode block (independent of problem size). Loading
`RecursiveFactorization` (e.g. for `RFLUFactorization`, the default dense
LU) additionally routes the dense panel kernels through
RecursiveFactorization/TriangularSolve automatically.

## Keyword Arguments

  - `reuse_symbolic`: reuse the cached symbolic analysis (and all numeric
    storage) across solves when the sparsity pattern is unchanged. Defaults
    to `true`.
  - `check_pattern`: check whether the sparsity pattern changed before
    reusing the analysis. Defaults to `true`.
  - `ordering`: fill-reducing ordering, `:amd` (default), `:nd`
    (pure-Julia nested dissection — best for large 3D meshes), or
    `:natural`.
  - `matching`: maximum-weight matching + scaling preprocessing, `:auto`
    (default — enabled when the diagonal is structurally weak), `true`, or
    `false`.
  - `eps_pivot`: static pivoting perturbation threshold (relative to
    `‖A‖`). Defaults to `1e-8`.
  - `threaded`: opt-in supernodal-elimination-tree parallel factorization
    (uses `Threads.nthreads()` tasks plus BLAS threads on the tree top; do
    not run other BLAS work concurrently). Defaults to `false`.
  - `dense_alg`: the dense LU algorithm used for the supernode diagonal
    blocks, through LinearSolve's own dense `init`/`solve!` machinery.
    `nothing` (default) resolves each block through `LinearSolve.defaultalg`
    — `RFLUFactorization` when RecursiveFactorization is loaded, MKL/LAPACK/
    generic LU otherwise — so the sparse solver always shares the dense
    default's engine.
"""
Base.@kwdef struct SupernodalLUFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
    check_pattern::Bool = true
    ordering::Symbol = :amd
    matching::Union{Symbol, Bool} = :auto
    eps_pivot::Float64 = 1.0e-8
    threaded::Bool = false
    dense_alg::Union{Nothing, AbstractDenseFactorization} = nothing
end

function init_cacheval(
        alg::SupernodalLUFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

"""
`SparseColumnPivotedQRFactorization(; reuse_symbolic = true, ordering = :default)`

A pure-Julia, rank-revealing column-pivoted sparse QR factorization, provided by
[SparseColumnPivotedQR.jl](https://github.com/SciML/SparseColumnPivotedQR.jl). It
targets the same "small-to-medium sparse" niche as KLU does for LU (low
symbolic-phase overhead, no SuiteSparse dependency) while preserving the
rank-revealing guarantees of LAPACK's column-pivoted QR, so it handles
rectangular (least-squares) and rank-deficient systems.

`SparseColumnPivotedQRFactorization` is a hard dependency of LinearSolve and is
the default sparse QR: it is the QR choice for non-square sparse systems in the
default polyalgorithm and the fallback when the default sparse LU
([`PureKLUFactorization`](@ref)/UMFPACK) hits a (near-)singular matrix.

## Keyword Arguments

  - `reuse_symbolic`: reuse the cached symbolic factorization across solves when
    the sparsity pattern is unchanged. Defaults to `true`.
  - `ordering`: column ordering passed to `SparseColumnPivotedQR.scpqr`
    (`:default`, `:amd`, `:natural`). LinearSolve loads AMD alongside the sparse
    extension, so `:default` resolves to AMD ordering (1.5-2x faster factorization
    than `:natural`). Defaults to `:default`.
"""
Base.@kwdef struct SparseColumnPivotedQRFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool = true
    ordering::Symbol = :default
end

function init_cacheval(
        alg::SparseColumnPivotedQRFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

## CHOLMODFactorization

"""
`CHOLMODFactorization(; shift = 0.0, perm = nothing)`

A wrapper of CHOLMOD's polyalgorithm, mixing Cholesky factorization and ldlt.
Tries cholesky for performance and retries ldlt if conditioning causes Cholesky
to fail.

Only supports sparse matrices.

!!! note

    CHOLMOD expects a structurally symmetric/Hermitian sparse matrix. Wrap the
    input in `Symmetric(A)` or `Hermitian(A)` when the matrix is symmetric by
    construction.

## Keyword Arguments

  - shift: the shift argument in CHOLMOD.
  - perm: the perm argument in CHOLMOD
"""
Base.@kwdef struct CHOLMODFactorization{T} <: AbstractSparseFactorization
    shift::Float64 = 0.0
    perm::T = nothing
end

function init_cacheval(
        alg::CHOLMODFactorization,
        A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::CHOLMODFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)

    if cache.isfresh
        cacheval = @get_cacheval(cache, :CHOLMODFactorization)
        fact = cholesky(A; check = false)
        if !LinearAlgebra.issuccess(fact)
            ldlt!(fact, A; check = false)
        end
        cache.cacheval = fact
        cache.isfresh = false
    end

    cache.u .= @get_cacheval(cache, :CHOLMODFactorization) \ cache.b
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = ReturnCode.Success
    )
end

## NormalCholeskyFactorization

"""
`NormalCholeskyFactorization(pivot = RowMaximum())`

A fast factorization which uses a Cholesky factorization on A * A'. Can be much
faster than LU factorization, but is not as numerically stable and thus should only
be applied to well-conditioned matrices.

!!! warning

    `NormalCholeskyFactorization` should only be applied to well-conditioned matrices. As a
    method it is not able to easily identify possible numerical issues. As a check it is
    recommended that the user checks `A*u-b` is approximately zero, as this may be untrue
    even if `sol.retcode === ReturnCode.Success` due to numerical stability issues.

## Positional Arguments

  - pivot: Defaults to RowMaximum(), but can be NoPivot()
"""
struct NormalCholeskyFactorization{P} <: AbstractDenseFactorization
    pivot::P
end

function NormalCholeskyFactorization(; pivot = nothing)
    pivot === nothing && (pivot = NoPivot())
    return NormalCholeskyFactorization(pivot)
end

default_alias_A(::NormalCholeskyFactorization, ::Any, ::Any) = true
default_alias_b(::NormalCholeskyFactorization, ::Any, ::Any) = true

const PREALLOCATED_NORMALCHOLESKY = ArrayInterface.cholesky_instance(rand(1, 1), NoPivot())

function init_cacheval(
        alg::NormalCholeskyFactorization, A::SMatrix, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return cholesky(Symmetric((A)' * A))
end

function init_cacheval(
        alg::NormalCholeskyFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    A_ = convert(AbstractMatrix, A)
    return ArrayInterface.cholesky_instance(
        Symmetric(Matrix{eltype(A)}(undef, 0, 0)), _normalize_pivot(alg.pivot)
    )
end

const PREALLOCATED_NORMALCHOLESKY_SYMMETRIC = ArrayInterface.cholesky_instance(
    Symmetric(rand(1, 1)), NoPivot()
)

function init_cacheval(
        alg::NormalCholeskyFactorization, A::Matrix{Float64}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return PREALLOCATED_NORMALCHOLESKY_SYMMETRIC
end

function init_cacheval(
        alg::NormalCholeskyFactorization,
        A::Union{Diagonal, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(cache::LinearCache, alg::NormalCholeskyFactorization; kwargs...)
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        if issparsematrixcsc(A) || A isa GPUArraysCore.AnyGPUArray || A isa SMatrix
            fact = cholesky(Symmetric((A)' * A); check = false)
        else
            fact = cholesky(Symmetric((A)' * A), _normalize_pivot(alg.pivot); check = false)
        end
        cache.cacheval = fact

        if hasmethod(LinearAlgebra.issuccess, Tuple{typeof(fact)}) &&
                !LinearAlgebra.issuccess(fact)
            @SciMLMessage("Solver failed", cache.verbose, :solver_failure)
            return SciMLBase.build_linear_solution(
                alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
            )
        end

        cache.isfresh = false
    end
    if issparsematrixcsc(A)
        cache.u .= @get_cacheval(cache, :NormalCholeskyFactorization) \ (A' * cache.b)
        y = cache.u
    elseif A isa StaticArray
        cache.u = @get_cacheval(cache, :NormalCholeskyFactorization) \ (A' * cache.b)
        y = cache.u
    else
        y = ldiv!(cache.u, @get_cacheval(cache, :NormalCholeskyFactorization), A' * cache.b)
    end
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

## NormalBunchKaufmanFactorization

"""
`NormalBunchKaufmanFactorization(rook = false)`

A fast factorization which uses a BunchKaufman factorization on A * A'. Can be much
faster than LU factorization, but is not as numerically stable and thus should only
be applied to well-conditioned matrices.

## Positional Arguments

  - rook: whether to perform rook pivoting. Defaults to false.
"""
struct NormalBunchKaufmanFactorization <: AbstractDenseFactorization
    rook::Bool
end

function NormalBunchKaufmanFactorization(; rook = false)
    return NormalBunchKaufmanFactorization(rook)
end

default_alias_A(::NormalBunchKaufmanFactorization, ::Any, ::Any) = true
default_alias_b(::NormalBunchKaufmanFactorization, ::Any, ::Any) = true

function init_cacheval(
        alg::NormalBunchKaufmanFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
        assumptions::OperatorAssumptions
    )
    return ArrayInterface.bunchkaufman_instance(convert(AbstractMatrix, A))
end

function SciMLBase.solve!(
        cache::LinearCache, alg::NormalBunchKaufmanFactorization;
        kwargs...
    )
    A = cache.A
    A = convert(AbstractMatrix, A)
    if cache.isfresh
        fact = bunchkaufman(Symmetric((A)' * A), alg.rook)
        cache.cacheval = fact
        cache.isfresh = false
    end
    y = ldiv!(cache.u, @get_cacheval(cache, :NormalBunchKaufmanFactorization), A' * cache.b)
    return SciMLBase.build_linear_solution(
        alg, y, nothing, nothing;
        retcode = ReturnCode.Success
    )
end

## DiagonalFactorization

"""
`DiagonalFactorization()`

A special implementation only for solving `Diagonal` matrices fast.
"""
struct DiagonalFactorization <: AbstractDenseFactorization end

function init_cacheval(
        alg::DiagonalFactorization, A, b, u, Pl, Pr, maxiters::Int,
        abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function SciMLBase.solve!(
        cache::LinearCache, alg::DiagonalFactorization;
        kwargs...
    )
    A = convert(AbstractMatrix, cache.A)
    if cache.u isa Vector && cache.b isa Vector
        @simd ivdep for i in eachindex(cache.u)
            cache.u[i] = A.diag[i] \ cache.b[i]
        end
    else
        cache.u .= A.diag .\ cache.b
    end
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, nothing;
        retcode = ReturnCode.Success
    )
end

## SparspakFactorization is here since it's MIT licensed, not GPL

"""
`SparspakFactorization(reuse_symbolic = true)`

This is the translation of the well-known sparse matrix software Sparspak
(Waterloo Sparse Matrix Package), solving
large sparse systems of linear algebraic equations. Sparspak is composed of the
subroutines from the book "Computer Solution of Large Sparse Positive Definite
Systems" by Alan George and Joseph Liu. Originally written in Fortran 77, later
rewritten in Fortran 90. Here is the software translated into Julia.

The Julia rewrite is released  under the MIT license with an express permission
from the authors of the Fortran package. The package uses multiple
dispatch to route around standard BLAS routines in the case e.g. of arbitrary-precision
floating point numbers or ForwardDiff.Dual.
This e.g. allows for Automatic Differentiation (AD) of a sparse-matrix solve.
"""
struct SparspakFactorization <: AbstractSparseFactorization
    reuse_symbolic::Bool

    function SparspakFactorization(; reuse_symbolic = true, throwerror = true)
        ext = Base.get_extension(@__MODULE__, :LinearSolveSparspakExt)
        return if throwerror && ext === nothing
            error("SparspakFactorization requires that Sparspak is loaded, i.e. `using Sparspak`")
        else
            new(reuse_symbolic)
        end
    end
end

"""
`STRUMPACKFactorization(; use_initial_guess = false, options = String[], kwargs...)`

A sparse direct solver based on
[STRUMPACK](https://github.com/pghysels/STRUMPACK) via the
`LinearSolveSTRUMPACKExt` extension.

This wrapper targets the single-node (`MT`) sparse interface and currently supports
real sparse matrices (`AbstractSparseMatrixCSC{<:AbstractFloat}`), solving in
`Float64` precision.

Pass STRUMPACK runtime options through `options` to expose advanced functionality.

Convenience keyword arguments are provided for common low-rank/compression tuning
and are translated to STRUMPACK-style runtime options:
- `compression` -> `--sp_compression`
- `rel_tol` -> `--sp_rel_tol`
- `abs_tol` -> `--sp_abs_tol`
- `max_rank` -> `--sp_max_rank`
- `leaf_size` -> `--sp_leaf_size`
- `reordering` -> `--sp_reordering_method`
- `matching` -> `--sp_enable_matching`

Any unexposed or version-specific knobs can still be passed through `options`.

!!! note

    Using this solver requires:
    1. `using SparseArrays` (to enable sparse matrix support), and
    2. loading `STRUMPACK_jll` (for example `import STRUMPACK_jll`).
"""
struct STRUMPACKFactorization <: AbstractSparseFactorization
    use_initial_guess::Bool
    options::Vector{String}

    function _push_opt_pair!(opts::Vector{String}, key::String, value)
        push!(opts, key)
        push!(opts, string(value))
        return opts
    end

    function STRUMPACKFactorization(
            ; use_initial_guess = false,
            options = String[],
            compression = nothing,
            rel_tol = nothing,
            abs_tol = nothing,
            max_rank = nothing,
            leaf_size = nothing,
            reordering = nothing,
            matching = nothing,
            throwerror = true
        )
        ext = Base.get_extension(@__MODULE__, :LinearSolveSTRUMPACKExt)
        return if throwerror && (ext === nothing || !ext.strumpack_isavailable())
            error("STRUMPACKFactorization requires `using SparseArrays` and loading `STRUMPACK_jll` (for example `import STRUMPACK_jll`)")
        else
            rel_tol !== nothing && rel_tol < 0 && error("`rel_tol` must be non-negative")
            abs_tol !== nothing && abs_tol < 0 && error("`abs_tol` must be non-negative")
            max_rank !== nothing && max_rank < 1 && error("`max_rank` must be >= 1")
            leaf_size !== nothing && leaf_size < 1 && error("`leaf_size` must be >= 1")

            runtime_options = String.(options)

            compression !== nothing && _push_opt_pair!(runtime_options, "--sp_compression", compression)
            rel_tol !== nothing && _push_opt_pair!(runtime_options, "--sp_rel_tol", rel_tol)
            abs_tol !== nothing && _push_opt_pair!(runtime_options, "--sp_abs_tol", abs_tol)
            max_rank !== nothing && _push_opt_pair!(runtime_options, "--sp_max_rank", Int(max_rank))
            leaf_size !== nothing && _push_opt_pair!(runtime_options, "--sp_leaf_size", Int(leaf_size))
            reordering !== nothing &&
                _push_opt_pair!(runtime_options, "--sp_reordering_method", reordering)
            matching !== nothing &&
                _push_opt_pair!(runtime_options, "--sp_enable_matching", matching ? 1 : 0)

            new(use_initial_guess, runtime_options)
        end
    end
end

function init_cacheval(
        ::STRUMPACKFactorization,
        ::Union{AbstractMatrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        ::STRUMPACKFactorization, ::StaticArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        alg::SparspakFactorization,
        A::Union{AbstractMatrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        ::SparspakFactorization, ::StaticArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

## CliqueTreesFactorization is here since it's MIT licensed, not GPL

"""
    CliqueTreesFactorization(
        alg = nothing,
        snd = nothing,
        reuse_symbolic = true,
    )

The sparse Cholesky factorization algorithm implemented in CliqueTrees.jl.
The implementation is pure-Julia and accepts arbitrary numeric types. It is
somewhat slower than CHOLMOD.
"""
struct CliqueTreesFactorization{A, S} <: AbstractSparseFactorization
    alg::A
    snd::S
    reuse_symbolic::Bool

    function CliqueTreesFactorization(;
            alg::A = nothing,
            snd::S = nothing,
            reuse_symbolic = true,
            throwerror = true,
        ) where {A, S}

        ext = Base.get_extension(@__MODULE__, :LinearSolveCliqueTreesExt)

        return if throwerror && isnothing(ext)
            error("CliqueTreesFactorization requires that CliqueTrees is loaded, i.e. `using CliqueTrees`")
        else
            new{A, S}(alg, snd, reuse_symbolic)
        end
    end
end

function init_cacheval(
        ::CliqueTreesFactorization, ::Union{AbstractMatrix, Nothing, AbstractSciMLOperator}, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        ::CliqueTreesFactorization, ::StaticArray, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

# Fallback init_cacheval for extension-based algorithms when extensions aren't loaded
# These return nothing since the actual implementations are in the extensions
function init_cacheval(
        ::BLISLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        ::CudaOffloadLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

function init_cacheval(
        ::MetalLUFactorization, A, b, u, Pl, Pr,
        maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    return nothing
end

for alg in vcat(
        InteractiveUtils.subtypes(AbstractDenseFactorization),
        InteractiveUtils.subtypes(AbstractSparseFactorization)
    )
    @eval function init_cacheval(
            alg::$alg, A::MatrixOperator, b, u, Pl, Pr,
            maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
            assumptions::OperatorAssumptions
        )
        return init_cacheval(
            alg, A.A, b, u, Pl, Pr,
            maxiters::Int, abstol, reltol, verbose::Union{LinearVerbosity, Bool},
            assumptions::OperatorAssumptions
        )
    end
end
