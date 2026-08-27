abstract type _AdjointFactorizationReuse end
struct _DirectAdjointFactorizationReuse <: _AdjointFactorizationReuse end
struct _ExtractedAdjointFactorizationReuse <: _AdjointFactorizationReuse end
struct _NormalAdjointFactorizationReuse <: _AdjointFactorizationReuse end
struct _CustomAdjointFactorizationReuse <: _AdjointFactorizationReuse end
struct _NoAdjointFactorizationReuse <: _AdjointFactorizationReuse end
struct _UnspecifiedAdjointFactorizationReuse <: _AdjointFactorizationReuse end

_adjoint_factorization_reuse(::Type{<:SciMLLinearSolveAlgorithm}) =
    _NoAdjointFactorizationReuse()
_adjoint_factorization_reuse(::Type{<:AbstractFactorization}) =
    _UnspecifiedAdjointFactorizationReuse()

for Alg in (
        LUFactorization,
        GenericLUFactorization,
        GESVFactorization,
        QRFactorization,
        CholeskyFactorization,
        LDLtFactorization,
        SVDFactorization,
        BunchKaufmanFactorization,
        GenericFactorization,
        UMFPACKFactorization,
        KLUFactorization,
        PureKLUFactorization,
        CHOLMODFactorization,
        CliqueTreesFactorization,
        RFLUFactorization,
        MKLLUFactorization,
        MetalLUFactorization,
        MetalOffload32MixedLUFactorization,
        MKL32MixedLUFactorization,
        AppleAccelerate32MixedLUFactorization,
        OpenBLAS32MixedLUFactorization,
        RF32MixedLUFactorization,
    )
    @eval _adjoint_factorization_reuse(::Type{<:$Alg}) =
        _DirectAdjointFactorizationReuse()
end

for Alg in (
        AppleAccelerateLUFactorization,
        OpenBLASLUFactorization,
        BLISLUFactorization,
        FastLUFactorization,
        FastQRFactorization,
    )
    @eval _adjoint_factorization_reuse(::Type{<:$Alg}) =
        _ExtractedAdjointFactorizationReuse()
end

for Alg in (NormalCholeskyFactorization, NormalBunchKaufmanFactorization)
    @eval _adjoint_factorization_reuse(::Type{<:$Alg}) =
        _NormalAdjointFactorizationReuse()
end

for Alg in (
        SimpleLUFactorization,
        SparseColumnPivotedQRFactorization,
        ButterflyFactorization,
        PardisoJL,
        MUMPSFactorization,
        HSLMA57Factorization,
        HSLMA97Factorization,
    )
    @eval _adjoint_factorization_reuse(::Type{<:$Alg}) =
        _CustomAdjointFactorizationReuse()
end

# These integrations either do not cache a numeric factorization or their public
# backend API does not expose an adjoint solve using the cached factorization.
# Generic reverse paths preserve a copy of `A` and factorize `adjoint(A)`;
# AD backends without that fallback report the algorithm as unsupported.
for Alg in (
        DiagonalFactorization,
        PureUMFPACKFactorization,
        SupernodalLUFactorization,
        SparspakFactorization,
        STRUMPACKFactorization,
        CudaOffloadLUFactorization,
        CUDAOffload32MixedLUFactorization,
        CudaOffloadQRFactorization,
        CudaOffloadFactorization,
        AMDGPUOffloadLUFactorization,
        AMDGPUOffloadQRFactorization,
        CUSOLVERRFFactorization,
        ParUFactorization,
        SuperLUDISTFactorization,
        ElementalJL,
        SpecializedLUFactorization,
        SpecializedQRFactorization,
    )
    @eval _adjoint_factorization_reuse(::Type{<:$Alg}) =
        _NoAdjointFactorizationReuse()
end

function _standard_cache_factorization(cacheval)
    if cacheval isa Factorization
        return cacheval
    elseif cacheval isa _GenericLUFactorizationCache
        return cacheval.fact
    elseif cacheval isa Tuple && !isempty(cacheval) && first(cacheval) isa Factorization
        return first(cacheval)
    else
        return nothing
    end
end

_custom_cache_factorization(::AbstractFactorization, cacheval) = nothing

"""
    _cache_factorization(alg, cacheval)

Return the factorization exposed by an explicitly opted-in algorithm, or
`nothing`. The direct-cache fallback accepts a `LinearAlgebra.Factorization`
stored either directly or first in a tuple, but it is only used for algorithms
listed as `_DirectAdjointFactorizationReuse`. Custom cache layouts opt in with
`_custom_cache_factorization`.
"""
function _cache_factorization(alg::AbstractFactorization, cacheval)
    reuse = _adjoint_factorization_reuse(typeof(alg))
    return _cache_factorization(reuse, alg, cacheval)
end
_cache_factorization(::SciMLLinearSolveAlgorithm, cacheval) = nothing
_cache_factorization(::_DirectAdjointFactorizationReuse, alg, cacheval) =
    _standard_cache_factorization(cacheval)
_cache_factorization(::_ExtractedAdjointFactorizationReuse, alg, cacheval) =
    _custom_cache_factorization(alg, cacheval)
_cache_factorization(::_AdjointFactorizationReuse, alg, cacheval) = nothing

function _can_reuse_cache_factorization(alg::AbstractFactorization, cacheval)
    reuse = _adjoint_factorization_reuse(typeof(alg))
    return _can_reuse_cache_factorization(reuse, alg, cacheval)
end
_can_reuse_cache_factorization(::SciMLLinearSolveAlgorithm, cacheval) = false
_can_reuse_cache_factorization(::_DirectAdjointFactorizationReuse, alg, cacheval) =
    _standard_cache_factorization(cacheval) !== nothing
_can_reuse_cache_factorization(::_ExtractedAdjointFactorizationReuse, alg, cacheval) =
    _custom_cache_factorization(alg, cacheval) !== nothing
_can_reuse_cache_factorization(::_NormalAdjointFactorizationReuse, alg, cacheval) =
    _standard_cache_factorization(cacheval) !== nothing
_can_reuse_cache_factorization(::_CustomAdjointFactorizationReuse, alg, cacheval) =
    _custom_can_reuse_adjoint_factorization(alg, cacheval)
_can_reuse_cache_factorization(::_AdjointFactorizationReuse, alg, cacheval) = false

_custom_can_reuse_adjoint_factorization(::AbstractFactorization, cacheval) = false
_custom_adjoint_factorization_solve(::AbstractFactorization, cacheval, A, b) = nothing

"""
    _adjoint_factorization_solve(alg, cacheval, A, b)

Solve `adjoint(A) * x = b` using `alg`'s cached factorization, returning
`nothing` when that algorithm has not opted into reverse-pass reuse. Algorithms
whose cache does not directly represent `A` provide a solver-specific method.
"""
function _adjoint_factorization_solve(
        alg::AbstractFactorization, cacheval, A, b
    )
    reuse = _adjoint_factorization_reuse(typeof(alg))
    return _adjoint_factorization_solve(reuse, alg, cacheval, A, b)
end
_adjoint_factorization_solve(::SciMLLinearSolveAlgorithm, cacheval, A, b) = nothing

function _adjoint_factorization_solve(
        ::Union{_DirectAdjointFactorizationReuse, _ExtractedAdjointFactorizationReuse},
        alg, cacheval, A, b
    )
    factorization = _cache_factorization(alg, cacheval)
    return factorization === nothing ? nothing : factorization' \ b
end

function _adjoint_factorization_solve(
        ::_NormalAdjointFactorizationReuse, alg, cacheval, A, b
    )
    factorization = _standard_cache_factorization(cacheval)
    return factorization === nothing ? nothing : A * (factorization \ b)
end

function _adjoint_factorization_solve(
        ::_CustomAdjointFactorizationReuse, alg, cacheval, A, b
    )
    return _custom_adjoint_factorization_solve(alg, cacheval, A, b)
end

_adjoint_factorization_solve(::_AdjointFactorizationReuse, alg, cacheval, A, b) =
    nothing

function _adjoint_krylov_solve(
        alg::AbstractKrylovSubspaceMethod, A, b; abstol, reltol, verbose,
        Pl = nothing, Pr = nothing
    )
    invprob = LinearProblem(adjoint(A), b)
    return solve(invprob, alg; abstol, reltol, verbose, Pl, Pr).u
end

_custom_can_reuse_adjoint_factorization(::SimpleLUFactorization, ::LUSolver) = true

function _custom_adjoint_factorization_solve(
        ::SimpleLUFactorization, factorization::LUSolver, A, b::AbstractVector
    )
    n = factorization.n
    T = promote_type(eltype(factorization.A), eltype(b))
    y = similar(b, T, n)
    z = similar(b, T, n)
    x = similar(b, T, n)

    @inbounds for i in 1:n
        value = b[i]
        for j in 1:(i - 1)
            value -= conj(factorization.A[j, i]) * y[j]
        end
        y[i] = value / conj(factorization.A[i, i])
    end

    @inbounds for i in n:-1:1
        value = y[i]
        for j in (i + 1):n
            value -= conj(factorization.A[j, i]) * z[j]
        end
        z[i] = value
    end

    @inbounds for i in 1:n
        x[factorization.perms[i]] = z[i]
    end
    return x
end

# `LHLFactorization` reuses its reduction on the reverse pass. The system is `σI + τJ` and
# the cache already holds `J = ZHZ⁻¹` and the LU of the shifted Hessenberg for the current
# shift, so `Aᴴ x = b` is `x = Z⁻ᴴ(σI+τH)⁻ᴴZᴴ b` — the same three `O(n²)` phases as the
# forward solve, no refactorization (`lhl_ldivH!`), with the forward's iterative refinement
# mirrored against `Aᴴ` (`lhl_refineH!`). Only the dense workspace has these adjoint
# kernels; the block-triangular sparse factorization has none, so its adjoint is solved
# against the assembled system with a fresh sparse LU rather than reused.
_adjoint_factorization_reuse(::Type{<:LHLFactorization}) =
    _CustomAdjointFactorizationReuse()

_custom_can_reuse_adjoint_factorization(::LHLFactorization, c::LHLCache) =
    c.ws isa LHLWorkspace
_custom_can_reuse_adjoint_factorization(::LHLFactorization, ::Any) = false

# `σI + τJ` assembled — the adjoint the reverse pass forms its residual against. A
# `WOperator`'s own `adjoint` does not track `update_gamma!` (its forward `mul!` does, but
# the adjoint keeps the γ it was built with), so refinement must not lean on `A'`; and the
# generic reverse fallback cannot factorize an `adjoint(WOperator)` at all, so the sparse
# branch assembles the matrix here too.
_lhl_system_matrix(σ, τ, J) = σ * I + τ * J

function _custom_adjoint_factorization_solve(alg::LHLFactorization, c::LHLCache, A, b)
    (A isa AbstractMatrix || A isa WOperator) || return nothing
    σ, τ = _lhl_shift_pair(A)
    J = _lhl_jacobian(A)
    if c.ws isa LHLWorkspace
        ws = _lhl_sync!(c, A, alg, false)              # match the reduction and shift to A
        ws.info == 0 || return nothing                 # singular shift: let the caller report it
        T = promote_type(eltype(b), typeof(ws.σ))      # complex when the shift or b is
        x = copyto!(similar(b, T), b)
        M = alg.refine > 0 ? _lhl_system_matrix(σ, τ, J) : nothing
        return _lhl_adjoint_reuse_solve!(x, M, b, ws, alg.refine)
    end
    # sparse LHL has no adjoint reduction: solve the adjoint of the assembled system.
    return adjoint(_lhl_system_matrix(σ, τ, J)) \ b
end

function _lhl_adjoint_reuse_solve!(x::AbstractVector, M, b, ws, refine::Int)
    lhl_ldivH!(x, ws)
    refine > 0 && lhl_refineH!(x, M, b, ws, refine)
    return x
end

# A batched (matrix) right-hand side solves column by column against the one reduction.
function _lhl_adjoint_reuse_solve!(x::AbstractMatrix, M, b, ws, refine::Int)
    for j in axes(x, 2)
        xj = view(x, :, j)
        lhl_ldivH!(xj, ws)
        refine > 0 && lhl_refineH!(xj, M, view(b, :, j), ws, refine)
    end
    return x
end

"""
    _adjoint_solve(cache::LinearCache, b)

Solve `adjoint(A) x = b` for the cache's current `A`, reusing the factorization the
cache already holds when the algorithm has opted into that, and falling back to the
Krylov, default-solver and generic routes otherwise.

This is the one place that decides how an adjoint solve is carried out, so the public
`solve!(cache; adjoint = true)` and the reverse-mode rules take the same path.
"""
function _adjoint_solve(cache::LinearCache, b, A = cache.A)
    alg = cache.alg
    cached = _adjoint_factorization_solve(alg, cache.cacheval, A, b)
    cached === nothing || return cached

    if alg isa AbstractKrylovSubspaceMethod
        Pl, Pr = _adjoint_precs(alg, cache.sensealg, cache.Pl, cache.Pr)
        return _adjoint_krylov_solve(
            alg, A, b; cache.abstol, cache.reltol, cache.verbose, Pl, Pr
        )
    elseif alg isa DefaultLinearSolver
        return defaultalg_adjoint_eval(cache, b)
    end

    # No cached factorization to reuse, so factorize the adjoint directly. `A` is a
    # parameter because a factorization may have overwritten `cache.A`, and the reverse
    # rules keep the copy that was preserved before that happened.
    invprob = LinearProblem(adjoint(A), b)
    return solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
end
