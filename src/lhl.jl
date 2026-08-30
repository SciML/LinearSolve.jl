"""
    LHLFactorization(; balance = true, refine = 1)

Reduce the Jacobian once to upper Hessenberg form by Gaussian similarity transformations
with partial pivoting — the *LHL factorization*, see
[LHLFactorization.jl](https://github.com/SciML/LHLFactorization.jl) —

    J = Z H Z⁻¹,   Z = D·P·L

with `L` unit lower triangular (multipliers bounded by 1), `P` a permutation and `D` the
balancing diagonal, and then solve

    (J - M/γ) x = b   as   x = Z (H - M/γ)⁻¹ Z⁻¹ b.

The shift is invisible to `Z`, so a **new `γ` costs `O(n²)`** (rebuild and re-factorize the
shifted Hessenberg) instead of the `O(n³)` of a fresh LU.  That is the point of the
algorithm: adaptive implicit ODE solvers change `γ = c·dt` every step while holding `J`
fixed for many steps.

Give it the system matrix unassembled, as a `SciMLOperators.WOperator` — the split
`J - M/γ` an implicit solver already builds — and move the shift with
[`update_gamma!`](@ref).  A `WOperator` whose Jacobian is a dense matrix or matrix-backed
operator is also what `defaultalg` selects this algorithm for, at sizes where the
reduction pays.  The mass matrix must be a multiple of `I`.

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
error).  With it the elimination is well behaved (`κ(L) ~ n^1.9`, Hessenberg growth ≤ 65
over ~22 stiff and adversarial families × `n ∈ {100, 400, 800}` × `γ ∈ [10⁻⁶, 10²]`); the
`κ(Z)` exposure that remains is carried by the balancing diagonal, and near-nilpotent or
badly scaled Jacobians lose 8–13 digits unrefined.

**One** step of fixed-precision iterative refinement is what closes that: it puts 633 of
636 cells of that study within 10× of LU's forward error, at a cost of ~1.4–1.6 solves.
Hence `refine = 1` by default.

!!! tip "Inside a Newton loop, consider `refine = 0`"
    An implicit ODE solver does on the order of fourteen solves per `γ`, so refinement is
    charged against exactly the quantity the algorithm is saving.  Newton is itself a
    correction loop whose convergence test governs the answer, so an inexact linear solve
    costs at most an extra iteration — measured on the Brusselator, `refine = 0` roughly
    halves the run time at the same accuracy.  The default stays at 1 because a bare
    linear solve has no outer loop to lean on; an integrator should opt out explicitly.

`balance = true` applies a Parlett–Reinsch diagonal similarity (exact powers of two)
before the reduction.

`thread = Val(true)` lets the reduction — the `O(n³)` part, and the only part big enough to
be worth splitting — run on Polyester threads when Polyester is loaded and
`Threads.nthreads() > 1`. It is deterministic: the result is bit-identical for any thread
count. The per-γ shift and the solves stay serial, being `O(n²)`.

## Keyword Arguments

  - `balance`: balance the Jacobian before reducing it. Default `true`.
  - `refine`: steps of iterative refinement applied to each solve. Default `1`.
  - `thread`: thread the reduction. Default `Val(true)`.
"""
struct LHLFactorization{T} <: AbstractDenseFactorization
    balance::Bool
    refine::Int
end

function LHLFactorization(;
        balance::Bool = true, refine::Int = 1, thread::Union{Bool, Val} = Val(true)
    )
    return LHLFactorization{_lhl_unwrap(thread)}(balance, refine)
end

_lhl_unwrap(::Val{T}) where {T} = T::Bool
_lhl_unwrap(t::Bool) = t
_lhl_thread(::LHLFactorization{T}) where {T} = Val(T)

default_alias_A(::LHLFactorization, ::Any, ::Any) = true

"""
    LHLCache

The cached reduction plus the identity of the Jacobian it was taken of.

`LHLWorkspace` deliberately records only *that* it holds a reduction, not whose — so the
consumer has to track that itself. Two independent things can invalidate a reduction and
each needs its own signal: swapping in a *different* matrix (caught by `===` on the
Jacobian) and writing into the *same* one (caught by `jacobian_stale`, since the object is
unchanged). Keying on only one of them silently reuses a stale reduction.

`jac` is typed rather than `Any`: the `LinearCache` is parameterized on `A`, so the
Jacobian's type is fixed for the life of the cache and only the `nothing` of the
not-yet-reduced state widens it.
"""
mutable struct LHLCache{WS, JT}
    ws::WS
    jac::Union{Nothing, JT}
    LHLCache{WS, JT}(ws, jac) where {WS, JT} = new{WS, JT}(ws, jac)
end

LHLCache(ws::WS, jac::JT) where {WS, JT} = LHLCache{WS, JT}(ws, jac)
LHLCache(ws, ::Type{JT}) where {JT} = LHLCache{typeof(ws), JT}(ws, nothing)

function init_cacheval(
        alg::LHLFactorization, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    (A isa AbstractMatrix || A isa WOperator) ||
        return LHLCache(LHLWorkspace{eltype(u)}(0), Nothing)
    J = _lhl_jacobian(A)
    if issparsematrixcsc(J)
        # The sparse extension chooses its per-block factorization during analysis.
        F = lhl(J; shift = _lhl_shift_eltype(A, u), thread = _lhl_unwrap(_lhl_thread(alg)))
        return LHLCache(F, typeof(J))
    end
    ws = LHLWorkspace{eltype(J)}(size(A, 1); shift = _lhl_shift_eltype(A, u))
    return LHLCache(ws, typeof(J))
end

_lhl_do_reduce!(ws::LHLWorkspace, J, alg::LHLFactorization) =
    lhl_reduce!(ws, J, alg.balance, _lhl_thread(alg))
_lhl_do_reduce!(F, J, ::LHLFactorization) = lhl!(F, J)

"""
    _lhl_shift_eltype(A, u) -> Type

Element type the shifted Hessenberg is held in.

A real Jacobian with a *complex* `gamma` — what RadauIIA wants — keeps a real reduction and
only the shift goes complex, which is the point of the split. So this is the Jacobian's own
element type unless the shift or the right-hand side is complex, in which case it is its
complex counterpart; those are the only two the workspace accepts.
"""
function _lhl_shift_eltype(A, u)
    T = eltype(_lhl_jacobian(A))
    wide = promote_type(T, _lhl_gamma_eltype(A), eltype(u))
    return wide <: Complex ? Complex{real(T)} : T
end

_lhl_gamma_eltype(W::WOperator) = typeof(W.gamma)
_lhl_gamma_eltype(A::AbstractMatrix) = eltype(A)

"""
    _lhl_jacobian(A)

The matrix the reduction is taken of. For a split `W = -M/γ + J` that is `J`; for a bare
matrix it is the matrix itself.
"""
_lhl_jacobian(W::WOperator) = _lhl_unwrap(W.J)
_lhl_jacobian(A::AbstractMatrix) = A
_lhl_unwrap(J::AbstractMatrix) = J
_lhl_unwrap(J::MatrixOperator) = convert(AbstractMatrix, J)

"""
    _lhl_shift_pair(A) -> (σ, τ)

The system matrix as `σI + τJ`. `WOperator` holds `-M/γ + J` with `M = λI`, so
`(σ, τ) = (-λ/γ, 1)`; a bare matrix is `(0, 1)`.
"""
function _lhl_shift_pair(W::WOperator)
    λ = _lhl_massmatrix_λ(W.mass_matrix)
    return (-λ / W.gamma, one(eltype(W)))
end
_lhl_shift_pair(A::AbstractMatrix) = (zero(eltype(A)), one(eltype(A)))

_lhl_massmatrix_λ(mm::UniformScaling) = mm.λ
_lhl_massmatrix_λ(mm::Number) = mm
function _lhl_massmatrix_λ(mm)
    throw(
        ArgumentError("LHLFactorization needs a mass matrix that is a multiple of I; got $(typeof(mm)). Reducing a general pencil to Hessenberg–triangular form is not implemented.")
    )
end

# A `WOperator` says when the contents of `J` moved, so `isfresh` — which is also raised
# for a mere change of `gamma` — must not be allowed to force a reduction there, or the
# algorithm loses its whole reason to exist. Identity covers the case the flag cannot: a
# different `J` altogether, whose flag may already have been cleared by someone else.
function _lhl_needs_reduce(c::LHLCache, A, isfresh::Bool)
    ws = c.ws
    lhl_isreduced(ws) || return true
    c.jac === _lhl_jacobian(A) || return true
    return _lhl_contents_moved(A, isfresh)
end

_lhl_contents_moved(W::WOperator, isfresh::Bool) = jacobian_stale(W)
_lhl_contents_moved(::AbstractMatrix, isfresh::Bool) = isfresh

# Clearing the flag claims the operator: a second cache sharing this `WOperator` would
# never see the update. `SciMLOperators.mark_jacobian_current!` documents the constraint.
_lhl_claim!(W::WOperator) = mark_jacobian_current!(W)
_lhl_claim!(::AbstractMatrix) = nothing

# The workspace's `setproperty!` forwards straight to `setfield!` without the conversion
# Julia's default does, so a real `τ` cannot be stored into a complex shift (which is
# exactly the real-J/complex-γ case) unless it is converted here.
function _lhl_load_shift!(ws, σ, τ)
    lhl_shift!(ws, σ, τ)
    TG = typeof(ws.σ)
    ws.σ = convert(TG, σ)
    ws.τ = convert(TG, τ)
    return ws
end

function _lhl_sync!(c::LHLCache, A, alg::LHLFactorization, isfresh::Bool)
    ws = c.ws
    σ, τ = _lhl_shift_pair(A)
    fresh_reduction = _lhl_needs_reduce(c, A, isfresh)
    if fresh_reduction
        J = _lhl_jacobian(A)
        _lhl_do_reduce!(ws, J, alg)
        c.jac = J
        _lhl_claim!(A)
    end
    if fresh_reduction || ws.σ != σ || ws.τ != τ
        _lhl_load_shift!(ws, σ, τ)
    end
    return ws
end

function SciMLBase.solve!(cache::LinearCache, alg::LHLFactorization; kwargs...)
    A = cache.A
    (A isa AbstractMatrix || A isa WOperator) || throw(
        ArgumentError("LHLFactorization requires a matrix or a WOperator, got $(typeof(A))")
    )
    c = LinearSolve.@get_cacheval(cache, :LHLFactorization)
    ws = c.ws
    # Sync unconditionally rather than only on `isfresh`: the checks are a pointer and a
    # scalar compare, free next to the solve, and it makes an in-place `J .= …` followed by
    # `mark_jacobian_updated!` enough on its own, with no `reinit!`.
    _lhl_sync!(c, A, alg, cache.isfresh)
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

Set the `gamma` of the cache's `WOperator` — making its system matrix `J - M/γ` — and make
the cached factorization current for it again.

With [`LHLFactorization`](@ref) this is the cheap path the algorithm exists for: `O(n²)`,
re-using the reduction of `J`.  With any other algorithm that can consume a `WOperator` it
simply invalidates the factorization, so the next `solve!` rebuilds from scratch — correct,
just not cheap.  Callers may therefore use it without checking which algorithm is in play.

Changing the *contents* of `J` is a separate event; announce it with
`SciMLOperators.mark_jacobian_updated!`.
"""
function update_gamma!(cache::LinearCache, γ)
    A = cache.A
    A isa WOperator || throw(
        ArgumentError("update_gamma! needs the cache's `A` to be a `WOperator`, got $(typeof(A))")
    )
    SciMLOperators.update_coefficients!(A; gamma = γ)
    # `cache.cacheval` directly, not `@get_cacheval`: the algorithm in play may be the
    # default solver, whose cacheval struct has no `LHLFactorization` slot.
    _update_gamma!(cache, cache.cacheval, A)
    return cache
end

_update_gamma!(cache::LinearCache, ::Any, ::WOperator) = (cache.isfresh = true)

function _update_gamma!(cache::LinearCache, c::LHLCache, A::WOperator)
    if cache.isfresh || _lhl_needs_reduce(c, A, cache.isfresh)
        cache.isfresh = true
        return cache
    end
    σ, τ = _lhl_shift_pair(A)
    _lhl_load_shift!(c.ws, σ, τ)
    return cache
end
