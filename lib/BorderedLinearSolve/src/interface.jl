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
A new `b`, `c` or `d` keeps a [`BorderingBLS`](@ref) factorization, and refactors a
[`DirectBLS`](@ref) cache, since they sit in its assembled matrix. Passing only `f` or
`g` keeps the factorization.

Assigning `cache.prob = prob` replaces the whole problem and always refactors.

The new parts may change the state's size or shape. A new size builds the cache's buffers
anew, as `init` does, and a new shape refactors.

An update that throws, also one rejected before it starts, leaves the cache with its old
problem. If the update would have refactored, or it began to refresh the cache, the next
`solve!` or update refactors from the problem's arrays as they are then.

Neither way of updating may change the type of a part, since the cache is typed for its
problem. A tuple border of another length, a wide border for a single one, or an `Int` for
a `Float64` throws an `ArgumentError`. Call `init` for such a problem instead, or give a
border whose width will change as a vector of states.
"""
function SciMLBase.reinit!(
        cache::Union{BorderedCache, DirectCache};
        J = nothing, b = nothing, c = nothing, d = nothing, f = nothing, g = nothing
    )
    refactor = _refactors(cache, J, b, c, d)
    # A part passed in may have changed in place, so the factors are stale from here on,
    # also if the new problem is rejected below.
    refactor && setfield!(cache, :stale, true)
    old = getfield(cache, :prob)
    prob = BorderedLinearProblem(
        something(J, old.J), something(b, old.b), something(c, old.c),
        something(d, old.d), something(f, old.f), something(g, old.g)
    )
    _install!(cache, prob, refactor)
    return nothing
end

# Whether a change of these parts invalidates the cache's factorization: that of `J`.
_refactors(cache, J, b, c, d) = !isnothing(J)

function Base.setproperty!(cache::Union{BorderedCache, DirectCache}, name::Symbol, x)
    if name === :prob
        # A new problem may carry a new `J`, or the same one changed in place, so the factors
        # are stale from here on, also if `x` is rejected.
        setfield!(cache, :stale, true)
        _install!(cache, x, true)
    else
        setfield!(cache, name, convert(fieldtype(typeof(cache), name), x))
    end
    return x
end

# Store `prob` in `cache` and refresh the cache for it. If the refresh throws, the old problem
# goes back in, and the cache stays stale until a `solve!` or update refreshes it in full.
function _install!(cache, prob::BorderedLinearProblem, refactor::Bool)
    _check_types(cache, prob)
    old = getfield(cache, :prob)
    setfield!(cache, :prob, prob)
    try
        # A matrix-free `J` acts on states of one shape, so a new shape refactors too.
        _sync!(cache, refactor || size(prob.f) != size(old.f))
    catch
        setfield!(cache, :prob, old)
        rethrow()
    end
    return nothing
end

# Refresh `cache` for its stored problem. A refresh that throws may leave factors of a `J`
# that has since changed in place, so the cache is stale from the start of a refresh until
# one finishes, and a stale cache refactors from `J` as it is at its next refresh.
function _sync!(cache, refactor::Bool)
    stale = getfield(cache, :stale)
    setfield!(cache, :stale, true)
    _refresh!(cache, refactor || stale)
    setfield!(cache, :stale, false)
    return nothing
end

# A cache's buffers are typed for the problem it was built for, so an update cannot change
# the type of any part, such as the length of a tuple border.
function _check_types(cache, prob::BorderedLinearProblem)
    P = fieldtype(typeof(cache), :prob)
    prob isa P && return nothing
    changed = [
        "$name from $(fieldtype(P, name)) to $(fieldtype(typeof(prob), name))"
            for name in fieldnames(P) if fieldtype(P, name) != fieldtype(typeof(prob), name)
    ]
    throw(
        ArgumentError(
            "an update cannot change the type of a part of the problem, but this one changes " *
                "$(join(changed, ", ")); call init with the new problem instead"
        )
    )
end
