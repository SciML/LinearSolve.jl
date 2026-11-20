"""
    BorderingBLS(; inner = nothing, k = 0, tol = 1e-12, det_sign = false)

Block elimination. Solves `J x₁ = f` and `J Δᵢ = bᵢ`, then

    solve (d - c'Δ) σ = (g - c'x₁)
    v = x₁ - Δ σ

Every inner solve uses the same `J`, so they share one `LinearSolve` cache; `inner` is
the `LinearSolve` algorithm for them (`nothing` picks the default).

`k` is the number of correction sweeps in the sense of Govaerts' BEC+k: after the first
pass the true residual of the bordered system is formed and used to solve for a
correction, which recovers accuracy when `J` is close to singular. That happens exactly
where continuation is interesting, near a fold, so it is worth having, but each sweep
costs another pass and the default of `0` skips it. A sweep is skipped early once the
residual is below `tol`.

The solve fails when an inner solve fails or `σ` is not finite. A zero pivot in the Schur
complement `d - c'Δ` gives such a `σ`, as does one so small that `σ` overflows. As with
`LUFactorization`, a singular `d - c'Δ` that rounding leaves with a small nonzero pivot
still succeeds.

Only `dot`, scaling and addition are used on the state, so the state can be any array,
not just a vector.

With `det_sign = true`, each successful `solve!` also keeps the sign of the bordered
determinant for [`det_sign`](@ref). `inner` must then be `LUFactorization()`.

# Reference

Govaerts, W. "Stable Solvers and Block Elimination for Bordered Systems." SIAM Journal
on Matrix Analysis and Applications 12, no. 3 (1991): 469-83.
"""
Base.@kwdef struct BorderingBLS{S, T <: Real}
    inner::S = nothing
    k::Int = 0
    tol::T = 1.0e-12
    det_sign::Bool = false
end

BorderingBLS(inner) = BorderingBLS(; inner)

"""
    BorderedCache

Holds the `LinearSolve` cache for `J` across solves. Reusing it is the whole reason to
prefer this over assembling the bordered matrix: the factorization or Krylov workspace
built for `J` is kept. After updating `J` in place between outer steps, pass it to
[`reinit!`](@ref) so the next `solve!` refactors.
"""
mutable struct BorderedCache{A, C, P, K, T}
    alg::A
    inner::C
    prob::P
    # The `init` keywords, kept to build a new inner cache for a new state size.
    kwargs::K
    # Whether `inner` may not match `prob`, after a refresh that threw; see `_sync!`.
    stale::Bool
    # What `det_sign` returns: the sign from the last `solve!`, or `nothing` if none is held.
    determinant_sign::Union{Nothing, T}
end

function SciMLBase.init(prob::BorderedLinearProblem, alg::BorderingBLS; kwargs...)
    _check_signing(alg)
    inner = _inner_cache(prob, alg, kwargs)
    # The sign of `det J` takes the type of `J`, and the Schur complement also that of `c` and `d`.
    T = float(promote_type(eltype(inner.A), eltype(inner.u), eltype(prob.d), map(eltype, _border_parts(prob.c))...))
    return BorderedCache{typeof(alg), typeof(inner), typeof(prob), typeof(kwargs), T}(alg, inner, prob, kwargs, false, nothing)
end

function _inner_cache(prob::BorderedLinearProblem, alg::BorderingBLS, kwargs)
    # A state that is not a vector is flattened for the inner solve only; see
    # flatten.jl. The bordering algebra below always works in the state's own shape.
    op = _flatten_operator(prob.J, prob.f)
    return init(LinearProblem(op, _rhs_copy(_flatten(prob.f, prob.f))), alg.inner; kwargs...)
end

# The refresh behind `reinit!` and `cache.prob = prob`. A new state size gets a new inner cache,
# built as at `init`. Otherwise `refactor` hands the inner cache `J` again, copied unless
# `alias_A` allows otherwise: most dense factorizations overwrite `A`, and BEC+k re-reads `J`.
function _refresh!(cache::BorderedCache, refactor::Bool)
    prob = cache.prob
    if length(prob.f) != length(cache.inner.b)
        cache.inner = _inner_cache(prob, cache.alg, cache.kwargs)
    elseif refactor
        op = _flatten_operator(prob.J, prob.f)
        cache.inner.A = op isa AbstractMatrix && !cache.inner.alias_A ? copy(op) : op
    end
    return nothing
end

# The right-hand side is copied in. Some backends solve in place and overwrite `b`,
# which would both mutate the problem the caller handed us and corrupt the residual the
# correction sweeps re-read.
_rhs_copy(b) = copy(b)

function _solve_with(cache::BorderedCache, rhs)
    f = cache.prob.f
    cache.inner.b = _rhs_copy(_flatten(rhs, f))
    sol = solve!(cache.inner)
    return _unflatten(copy(sol.u), f), SciMLBase.successful_retcode(sol.retcode)
end

# One block-elimination pass. `f` and `g` are explicit so the correction sweeps can
# reuse this with the residual in place of the original right-hand side.
function _bec(cache::BorderedCache, f, g)
    prob = cache.prob
    x1, ok = _solve_with(cache, f)

    parts = _border_parts(prob.b)
    Δ = map(parts) do bi
        δ, okᵢ = _solve_with(cache, bi)
        ok &= okᵢ
        δ
    end

    σ, schur_sign = _border_solve(prob, x1, Δ, g, cache.alg.det_sign)
    v = x1 - _combine(Δ, σ)
    # A zero pivot in the Schur complement leaves `σ` infinite or `NaN`, as does a tiny one.
    return v, σ, schur_sign, ok && all(isfinite, σ) ? ReturnCode.Success : ReturnCode.Failure
end

# `c'x` over states of any shape.
_cdot(c, x) = dot(c, x)

# `σ`, and the sign of the determinant of the Schur complement `S = d - c'Δ` it solves
# with where `signing` asks for it. A zero pivot in `S` gives a `σ` that is not finite.
function _border_solve(prob, x1, Δ, g, signing::Bool)
    cs = _border_parts(prob.c)
    if _border_width(prob.b) == 1
        # Scalar division, which stays on whatever device the state lives on.
        S = prob.d - _cdot(only(cs), only(Δ))
        return (g - _cdot(only(cs), x1)) / S, sign(S)
    end
    # A dense m x m solve. `m` is small by assumption, and doing it on the host keeps
    # it off a GPU `\`, whose pivoting would scalar-index at these sizes.
    m = length(cs)
    S = Array(prob.d) - [_cdot(cs[i], Δ[j]) for i in 1:m, j in 1:m]
    rhs = Array(g) - [_cdot(cs[i], x1) for i in 1:m]
    try
        # `\` may solve a structured `S` without an LU; only the sign needs one.
        signing || return S \ rhs, nothing
        factors = lu(S)
        return factors \ rhs, last(logabsdet(factors))
    catch err
        err isa SingularException || rethrow()
        # Where the width-1 division gives `Inf` or `NaN`, `\` and `lu` throw. Fail the same way.
        return fill(float(promote_type(eltype(S), eltype(rhs)))(NaN), m), nothing
    end
end

_combine(Δ, σ::Number) = only(Δ) * σ
_combine(Δ, σ::AbstractVector) = sum(Δ[i] * σ[i] for i in eachindex(Δ))

function SciMLBase.solve!(cache::BorderedCache{<:BorderingBLS})
    cache.determinant_sign = nothing
    cache.stale && _sync!(cache, true)
    alg, prob = cache.alg, cache.prob
    # The correction sweeps solve with the same `S`, so the first pass gives its sign.
    v, σ, schur_sign, rc = _bec(cache, prob.f, prob.g)

    for _ in 1:(alg.k)
        rf = prob.f - _apply(prob.J, v) - _combine(_border_parts(prob.b), σ)
        rg = prob.g - _cdot_all(prob.c, v) - _dmul(prob.d, σ)
        if max(norm(rf), norm(rg)) <= alg.tol
            break
        end
        dv, dσ, _, rc2 = _bec(cache, rf, rg)
        v = v + dv
        σ = σ + dσ
        SciMLBase.successful_retcode(rc2) || (rc = ReturnCode.Failure)
    end

    if alg.det_sign && SciMLBase.successful_retcode(rc)
        cache.determinant_sign = _lu_sign(cache.inner) * schur_sign
    end
    return BorderedLinearSolution(v, σ, rc)
end

_apply(J, v) = J * v
# A border of width one may be a single state, or a tuple or vector holding one.
_cdot_all(c, v) = _border_width(c) == 1 ? _cdot(only(_border_parts(c)), v) : [_cdot(ci, v) for ci in _border_parts(c)]
_dmul(d, σ) = d * σ
