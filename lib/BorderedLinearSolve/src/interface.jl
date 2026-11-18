"""
    solve(prob::BorderedLinearProblem, alg)

Solve the bordered system in one shot. Use `init` plus `solve!` instead when the
same `J` is solved against repeatedly, which is the case this package is for.
"""
SciMLBase.solve(prob::BorderedLinearProblem, alg; kwargs...) =
    solve!(SciMLBase.init(prob, alg; kwargs...))

SciMLBase.solve(prob::BorderedLinearProblem; kwargs...) =
    SciMLBase.solve(prob, BorderingBLS(); kwargs...)

"""
    reinit!(cache; J = nothing, b = nothing, c = nothing, d = nothing, f = nothing, g = nothing)

Replace the given parts of the cached problem and keep the rest, as `LinearSolve`'s
`reinit!` does for its own cache.

Passing `J` marks the factorization stale, also when it is the same array changed in
place, so the next `solve!` refactors. A cache cannot see an in-place change on its own:
without `reinit!`, `solve!` keeps solving against the factorization of the old `J`.
Passing only `b`, `c`, `d`, `f` or `g` keeps the factorization.

Assigning `cache.prob = prob` replaces the whole problem and always refactors.

The new parts may change the state size; the cache then builds its buffers anew, as `init`
does. An update that throws leaves the cache with its old problem, which the next `solve!`
refactors.
"""
function SciMLBase.reinit!(
        cache::Union{BorderedCache, DirectCache};
        J = nothing, b = nothing, c = nothing, d = nothing, f = nothing, g = nothing
    )
    old = getfield(cache, :prob)
    prob = BorderedLinearProblem(
        something(J, old.J), something(b, old.b), something(c, old.c),
        something(d, old.d), something(f, old.f), something(g, old.g)
    )
    _install!(cache, prob, !isnothing(J))
    return nothing
end

function Base.setproperty!(cache::Union{BorderedCache, DirectCache}, name::Symbol, x)
    converted = convert(fieldtype(typeof(cache), name), x)
    # A new problem may carry a new `J`, or the same one changed in place.
    name === :prob ? _install!(cache, converted, true) : setfield!(cache, name, converted)
    return x
end

# Store `prob` in `cache` and refresh the cache for it. A refresh that throws may have
# overwritten part of the old factorization, so the old problem goes back in and is
# refreshed in full: a stored problem never meets a factorization of another.
function _install!(cache, prob, refactor::Bool)
    old = getfield(cache, :prob)
    setfield!(cache, :prob, prob)
    try
        _refresh!(cache, refactor)
    catch
        setfield!(cache, :prob, old)
        _refresh!(cache, true)
        rethrow()
    end
    return nothing
end
