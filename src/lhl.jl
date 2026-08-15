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
[`update_gamma!`](@ref).  A `WOperator` whose Jacobian is a dense matrix is also what
`defaultalg` selects this algorithm for, at sizes where the reduction pays.  The mass
matrix must be a multiple of `I`.

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
    (A isa AbstractMatrix || A isa WOperator) || return LHLWorkspace{eltype(u)}(0)
    return LHLWorkspace{eltype(A)}(size(A, 1))
end

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

# A `WOperator` says exactly when `J` moved, so `isfresh` — which is also raised for a mere
# change of `gamma` — must not be allowed to force a reduction. A bare matrix has no such
# signal, so there `isfresh` is all there is.
_lhl_needs_reduce(ws::LHLWorkspace, W::WOperator, isfresh::Bool) =
    ws.jac_version != jacobian_version(W) || ws.n != size(W, 1)
_lhl_needs_reduce(ws::LHLWorkspace, A::AbstractMatrix, isfresh::Bool) =
    isfresh || ws.jac_version < 0 || ws.n != size(A, 1)

_lhl_stamp!(ws::LHLWorkspace, W::WOperator) = (ws.jac_version = jacobian_version(W))
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
    (A isa AbstractMatrix || A isa WOperator) || throw(
        ArgumentError("LHLFactorization requires a matrix or a WOperator, got $(typeof(A))")
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

function _update_gamma!(cache::LinearCache, ws::LHLWorkspace, A::WOperator)
    if cache.isfresh || _lhl_needs_reduce(ws, A, cache.isfresh)
        cache.isfresh = true
        return cache
    end
    σ, τ = _lhl_shift_pair(A)
    lhl_shift!(ws, σ, τ)
    ws.σ, ws.τ = σ, τ
    return cache
end
