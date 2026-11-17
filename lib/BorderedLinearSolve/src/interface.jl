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
    setfield!(cache, :prob, prob)
    isnothing(J) || _refresh!(cache)
    return nothing
end

function Base.setproperty!(cache::Union{BorderedCache, DirectCache}, name::Symbol, x)
    setfield!(cache, name, convert(fieldtype(typeof(cache), name), x))
    # A new problem may carry a new `J`, or the same one changed in place.
    name === :prob && _refresh!(cache)
    return x
end
