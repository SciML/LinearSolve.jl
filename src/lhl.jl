"""
    LHLFactorization(; balance = true, refine = 1)

Reduce the Jacobian once to upper Hessenberg form by Gaussian similarity transformations
with partial pivoting — the *LHL factorization*, see [LHL.jl](https://github.com/SciML/LHL.jl) —

    J = Z H Z⁻¹,   Z = D·P·L

with `L` unit lower triangular (multipliers bounded by 1), `P` a permutation and `D` the
balancing diagonal, and then solve

    (I - γJ) x = b   as   x = Z (I - γH)⁻¹ Z⁻¹ b.

The shift is invisible to `Z`, so a **new `γ` costs `O(n²)`** (rebuild and re-factorize the
Hessenberg `I - γH`) instead of the `O(n³)` of a fresh LU.  That is the point of the
algorithm: adaptive implicit ODE solvers change `γ = c·dt` every step while holding `J`
fixed for many steps.  Give it a [`ShiftedJacobian`](@ref) as the system matrix and change
`γ` with [`update_gamma!`](@ref); this is also what `defaultalg` selects for a
`ShiftedJacobian` large enough to pay for the reduction.

Handed an ordinary matrix `A`, it solves `Ax = b` as `Z H⁻¹ Z⁻¹ b`; that works, but it is
strictly worse than an LU and the algorithm has no reason to be chosen.

## Cost

| | reduction | new `γ` | solve |
|---|---|---|---|
| `LUFactorization` | — | ⅔n³ | 2n² |
| `LHLFactorization` | 5/3n³ | n² | 3n² |

## Stability

`Z` is not orthogonal, so unlike an LU the backward error carries a factor `κ(Z)`.
Partial pivoting bounds the multipliers by 1 and is **not optional** — without it the
factorization loses all accuracy on ordinary matrices (measured: `O(10⁻¹)` backward
error).  With it, `κ(Z)` is a few thousand on typical Jacobians, but matrices whose
leading Krylov basis is nearly rank deficient (near-nilpotent, tightly clustered spectra)
can push it to `10¹⁰` and cost eight digits.  `refine` steps of fixed-precision iterative
refinement (default 1, each `O(n²)`) restore a backward error comparable to LU's.

!!! tip "Inside a Newton loop, use `refine = 0`"
    Refinement roughly triples the cost of a solve, and an implicit ODE solver does on the
    order of fourteen solves per `γ` — so it is charged against exactly the quantity the
    algorithm is trying to save.  A Newton iteration is itself a correction loop, and its
    convergence test governs the answer, so an inexact linear solve costs at most an extra
    iteration.  On the Brusselator, `refine = 0` halved the run time relative to
    `refine = 1` at the same accuracy; `refine = 1` remains the default because a bare
    linear solve has no outer loop to lean on.

`balance = true` applies a Parlett–Reinsch diagonal similarity (exact powers of two)
before the reduction.

## Keyword Arguments

  - `balance`: balance the Jacobian before reducing it. Default `true`.
  - `refine`: steps of iterative refinement applied to each solve. Default `1`.
"""
struct LHLFactorization <: AbstractDenseFactorization
    balance::Bool
    refine::Int
end

LHLFactorization(; balance::Bool = true, refine::Int = 1) = LHLFactorization(balance, refine)

default_alias_A(::LHLFactorization, ::Any, ::Any) = true

function init_cacheval(
        alg::LHLFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    A isa AbstractMatrix || return LHLWorkspace{eltype(u)}(0)
    return LHLWorkspace{eltype(A)}(size(A, 1))
end

_lhl_jacobian(W::ShiftedJacobian) = W.J
_lhl_jacobian(A::AbstractMatrix) = A
_lhl_shift_pair(W::ShiftedJacobian) = (W.β, W.α)
_lhl_shift_pair(A::AbstractMatrix) = (zero(eltype(A)), one(eltype(A)))

# A `ShiftedJacobian` says exactly when `J` moved, so `isfresh` — which is also raised for
# a mere change of shift — must not be allowed to force a reduction.  A bare matrix has no
# such signal, so there `isfresh` is all there is.
_lhl_needs_reduce(ws::LHLWorkspace, W::ShiftedJacobian, isfresh::Bool) =
    ws.jac_version != W.jac_version || ws.n != size(W, 1)
_lhl_needs_reduce(ws::LHLWorkspace, A::AbstractMatrix, isfresh::Bool) =
    isfresh || ws.jac_version < 0 || ws.n != size(A, 1)

_lhl_stamp!(ws::LHLWorkspace, W::ShiftedJacobian) = (ws.jac_version = W.jac_version)
_lhl_stamp!(ws::LHLWorkspace, ::AbstractMatrix) = (ws.jac_version = 0)

function _lhl_sync!(ws::LHLWorkspace, A, alg::LHLFactorization, isfresh::Bool)
    σ, τ = _lhl_shift_pair(A)
    fresh_reduction = _lhl_needs_reduce(ws, A, isfresh)
    if fresh_reduction
        lhl_reduce!(ws, _lhl_jacobian(A), alg.balance)
        _lhl_stamp!(ws, A)
    end
    if fresh_reduction || ws.σ != σ || ws.τ != τ
        lhl_shift!(ws, σ, τ)
        ws.σ, ws.τ = σ, τ
    end
    return ws
end

function SciMLBase.solve!(cache::LinearCache, alg::LHLFactorization; kwargs...)
    A = cache.A
    A isa AbstractMatrix || throw(
        ArgumentError("LHLFactorization requires a matrix or a ShiftedJacobian, got $(typeof(A))")
    )
    ws = LinearSolve.@get_cacheval(cache, :LHLFactorization)
    # Sync unconditionally rather than only on `isfresh`: an integer and a scalar compare
    # are free next to the solve, and it makes an in-place `J .= …` followed by
    # `mark_jacobian_updated!` enough on its own, with no `reinit!`.
    _lhl_sync!(ws, A, alg, cache.isfresh)
    cache.isfresh = false
    if ws.info != 0
        return SciMLBase.build_linear_solution(
            alg, cache.u, nothing, nothing; retcode = ReturnCode.Failure
        )
    end
    y = cache.u
    copyto!(y, cache.b)
    lhl_ldiv!(y, ws)
    lhl_refine!(y, A, cache.b, ws, alg.refine)
    return SciMLBase.build_linear_solution(alg, y, nothing, nothing; retcode = ReturnCode.Success)
end

"""
    update_gamma!(cache::LinearCache, γ) -> cache

Set the shift of the cache's [`ShiftedJacobian`](@ref) to `γ` and make the cached
factorization current for it again.

With [`LHLFactorization`](@ref) this is the cheap path the algorithm exists for: `O(n²)`,
re-using the reduction of `J`.  With any other algorithm it simply invalidates the
factorization, so the next `solve!` refactorizes `I - γJ` from scratch — correct, just not
cheap.  Callers may therefore use it unconditionally.

Changing the *contents* of `J` is a separate event; announce it with
[`mark_jacobian_updated!`](@ref).
"""
update_gamma!(cache::LinearCache, γ) = update_shift!(cache, -γ, oneunit(γ))

"""
    update_shift!(cache::LinearCache, α, β) -> cache

The general form of [`update_gamma!`](@ref): make the cache's system matrix `α*J + β*I`.

`update_gamma!(cache, γ)` is `update_shift!(cache, -γ, 1)`; an implicit ODE solver using
the W-transform `W = J - M/(dt·γ)` wants `update_shift!(cache, 1, -inv(dt*γ))`.
"""
function update_shift!(cache::LinearCache, α, β)
    A = cache.A
    A isa ShiftedJacobian || throw(
        ArgumentError("update_gamma!/update_shift! need the cache's `A` to be a `ShiftedJacobian`, got $(typeof(A))")
    )
    set_shift!(A, α, β)
    _update_shift!(cache, LinearSolve.@get_cacheval(cache, :LHLFactorization), A)
    return cache
end

_update_shift!(cache::LinearCache, ::Any, ::ShiftedJacobian) = (cache.isfresh = true)

function _update_shift!(cache::LinearCache, ws::LHLWorkspace, A::ShiftedJacobian)
    if cache.isfresh || _lhl_needs_reduce(ws, A, cache.isfresh)
        cache.isfresh = true
        return cache
    end
    lhl_shift!(ws, A.β, A.α)
    ws.σ, ws.τ = A.β, A.α
    return cache
end
