"""
    DirectBLS(; inner = nothing)

Assemble the full bordered matrix and hand it to `LinearSolve`.

This gives up the cache reuse that [`BorderingBLS`](@ref) exists for, and for a sparse
`J` the assembled matrix has a dense row and column that the factorization has to carry.
It is here because it is the reference the bordering result should be checked against,
and because for a small `J` it is simply faster than repeated inner solves.

Assembling requires a `J` that is an actual matrix and a state that can be flattened,
so unlike [`BorderingBLS`](@ref) this is not available for a matrix-free `J`.
"""
Base.@kwdef struct DirectBLS{S}
    inner::S = nothing
end

struct DirectCache{A, P}
    alg::A
    prob::P
end

SciMLBase.init(prob::BorderedLinearProblem, alg::DirectBLS; kwargs...) =
    DirectCache(alg, prob)

function _assemble(prob::BorderedLinearProblem)
    m = _border_width(prob.b)
    bcols = reduce(hcat, [vec(x) for x in _border_parts(prob.b)])
    ccols = reduce(hcat, [vec(x) for x in _border_parts(prob.c)])
    d = m == 1 ? fill(prob.d, 1, 1) : Array(prob.d)
    return [Matrix(prob.J) bcols; ccols' d]
end

function SciMLBase.solve!(cache::DirectCache)
    prob = cache.prob
    m = _border_width(prob.b)
    rhs = vcat(vec(prob.f), m == 1 ? [prob.g] : collect(prob.g))
    sol = solve!(init(LinearProblem(_assemble(prob), rhs), cache.alg.inner))
    n = length(prob.f)
    # Give `v` the shape of the state back, so a 2D state comes out 2D.
    v = reshape(sol.u[1:n], size(prob.f))
    tail = sol.u[(n + 1):end]
    return BorderedLinearSolution(v, m == 1 ? tail[1] : tail, sol.retcode)
end
