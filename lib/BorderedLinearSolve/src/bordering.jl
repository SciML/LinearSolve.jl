"""
    BorderingBLS(; inner = nothing, k = 0, tol = 1e-12)

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

Only `dot`, scaling and addition are used on the state, so the state can be any array,
not just a vector.

# Reference

Govaerts, W. "Stable Solvers and Block Elimination for Bordered Systems." SIAM Journal
on Matrix Analysis and Applications 12, no. 3 (1991): 469-83.
"""
Base.@kwdef struct BorderingBLS{S, T <: Real}
    inner::S = nothing
    k::Int = 0
    tol::T = 1.0e-12
end

BorderingBLS(inner) = BorderingBLS(; inner)

"""
    BorderedCache

Holds the `LinearSolve` cache for `J` across solves. Reusing it is the whole reason to
prefer this over assembling the bordered matrix: the factorization or Krylov workspace
built for `J` is kept, and `J` can be updated in place between outer steps.
"""
mutable struct BorderedCache{A, C, P}
    alg::A
    inner::C
    prob::P
end

function SciMLBase.init(prob::BorderedLinearProblem, alg::BorderingBLS; kwargs...)
    # A state that is not a vector is flattened for the inner solve only; see
    # flatten.jl. The bordering algebra below always works in the state's own shape.
    op = _flatten_operator(prob.J, prob.f)
    inner = init(LinearProblem(op, _rhs_copy(_flatten(prob.f, prob.f))), alg.inner; kwargs...)
    return BorderedCache(alg, inner, prob)
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

    σ = _border_solve(prob, x1, Δ, g)
    v = x1 - _combine(Δ, σ)
    return v, σ, ok ? ReturnCode.Success : ReturnCode.Failure
end

# `c'x` over states of any shape.
_cdot(c, x) = dot(c, x)

function _border_solve(prob, x1, Δ, g)
    cs = _border_parts(prob.c)
    if _border_width(prob.b) == 1
        # Scalar division, which stays on whatever device the state lives on.
        return (g - _cdot(only(cs), x1)) / (prob.d - _cdot(only(cs), only(Δ)))
    end
    # A dense m x m solve. `m` is small by assumption, and doing it on the host keeps
    # it off a GPU `\`, whose pivoting would scalar-index at these sizes.
    m = length(cs)
    S = Array(prob.d) - [_cdot(cs[i], Δ[j]) for i in 1:m, j in 1:m]
    rhs = Array(g) - [_cdot(cs[i], x1) for i in 1:m]
    return S \ rhs
end

_combine(Δ, σ::Number) = only(Δ) * σ
_combine(Δ, σ::AbstractVector) = sum(Δ[i] * σ[i] for i in eachindex(Δ))

function SciMLBase.solve!(cache::BorderedCache{<:BorderingBLS})
    alg, prob = cache.alg, cache.prob
    v, σ, rc = _bec(cache, prob.f, prob.g)

    for _ in 1:(alg.k)
        rf = prob.f - _apply(prob.J, v) - _combine(_border_parts(prob.b), σ)
        rg = prob.g - _cdot_all(prob.c, v) - _dmul(prob.d, σ)
        if max(norm(rf), norm(rg)) <= alg.tol
            break
        end
        dv, dσ, rc2 = _bec(cache, rf, rg)
        v = v + dv
        σ = σ + dσ
        SciMLBase.successful_retcode(rc2) || (rc = ReturnCode.Failure)
    end

    return BorderedLinearSolution(v, σ, rc)
end

_apply(J, v) = J * v
_cdot_all(c, v) = _border_width(c) == 1 ? _cdot(c, v) : [_cdot(ci, v) for ci in _border_parts(c)]
_dmul(d, σ) = d * σ
