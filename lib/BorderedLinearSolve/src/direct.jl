"""
    DirectBLS(; inner = nothing, det_sign = false)

Assemble the full bordered matrix and hand it to `LinearSolve`.

The cache keeps the assembled matrix and the `LinearSolve` cache over it, so a new
right-hand side ([`reinit!`](@ref) with `f` or `g` alone) reuses the factorization. A `J`
or border passed to `reinit!` is written into the same matrix and refactored at the next
`solve!`. Pass it also after changing it in place, since the cache cannot see that change.

What this gives up is the reuse across a changing `J` that [`BorderingBLS`](@ref) exists
for, and for a sparse `J` the assembled matrix has a dense row and column that the
factorization has to carry. It is here because it is the reference the bordering result
should be checked against, and because for a small `J` it is simply faster than repeated
inner solves.

Assembling requires a `J` that is an actual matrix and a state that can be flattened,
so unlike [`BorderingBLS`](@ref) this is not available for a matrix-free `J`.

With `det_sign = true`, each successful `solve!` also keeps the sign of the bordered
determinant for [`det_sign`](@ref). `inner` must then be `LUFactorization()`.
"""
Base.@kwdef struct DirectBLS{S}
    inner::S = nothing
    det_sign::Bool = false
end

"""
    DirectCache

Holds the assembled bordered matrix, the stacked right-hand side, and the `LinearSolve`
cache over them. The matrix is rewritten in full whenever `J` or the border changes,
since a factorization done in place leaves it holding the factors. A new state size or
border width builds all three anew.
"""
mutable struct DirectCache{A, M, R, C, P, K, T}
    alg::A
    matrix::M
    rhs::R
    inner::C
    prob::P
    # The `init` keywords, kept to build the cache anew for a new state size or border width.
    kwargs::K
    # Whether a refresh threw since the last one that finished; see `_sync!`.
    stale::Bool
    # What `det_sign` returns: the sign from the last `solve!`, or `nothing` if none is held.
    determinant_sign::Union{Nothing, T}
end

function SciMLBase.init(prob::BorderedLinearProblem, alg::DirectBLS; kwargs...)
    _check_signing(alg)
    matrix, rhs, inner = _direct_parts(prob, alg, kwargs)
    return DirectCache{typeof.((alg, matrix, rhs, inner, prob, kwargs))..., float(eltype(matrix))}(
        alg, matrix, rhs, inner, prob, kwargs, false, nothing
    )
end

# The assembled matrix and stacked right-hand side of `prob`, and the `LinearSolve` cache over them.
function _direct_parts(prob::BorderedLinearProblem, alg::DirectBLS, kwargs)
    n, m = length(prob.f), _border_width(prob.b)
    matrix_type = promote_type(
        eltype(prob.J), map(eltype, _border_parts(prob.b))...,
        map(eltype, _border_parts(prob.c))..., eltype(prob.d)
    )
    matrix = _assemble!(Matrix{matrix_type}(undef, n + m, n + m), prob)
    rhs = _stack_rhs!(Vector{promote_type(eltype(prob.f), eltype(prob.g))}(undef, n + m), prob)
    return matrix, rhs, init(LinearProblem(matrix, rhs), alg.inner; kwargs...)
end

# Every entry of `matrix` is written, so it may hold anything before, factors included.
function _assemble!(matrix::AbstractMatrix, prob::BorderedLinearProblem)
    n, m = length(prob.f), _border_width(prob.b)
    J = _as_matrix(prob.J)
    # The problem checks `J` only for a vector state. Whatever the state, assembly needs n x n.
    size(J) == (n, n) || throw(
        DimensionMismatch("J has size $(size(J)), expected ($n, $n) to match the state")
    )
    copyto!(view(matrix, 1:n, 1:n), J)
    for (i, bi) in enumerate(_border_parts(prob.b))
        copyto!(view(matrix, 1:n, n + i), vec(bi))
    end
    for (i, ci) in enumerate(_border_parts(prob.c))
        view(matrix, n + i, 1:n) .= conj.(vec(ci))
    end
    if m == 1
        matrix[n + 1, n + 1] = prob.d
    else
        copyto!(view(matrix, (n + 1):(n + m), (n + 1):(n + m)), prob.d)
    end
    return matrix
end

_as_matrix(J::AbstractMatrix) = J
_as_matrix(J) = Matrix(J)

function _stack_rhs!(rhs::AbstractVector, prob::BorderedLinearProblem)
    n = length(prob.f)
    copyto!(view(rhs, 1:n), vec(prob.f))
    if _border_width(prob.b) == 1
        rhs[n + 1] = prob.g
    else
        copyto!(view(rhs, (n + 1):length(rhs)), prob.g)
    end
    return rhs
end

# The border sits in the assembled matrix, so a new one refactors too.
_refactors(::DirectCache, J, b, c, d) = !(isnothing(J) && isnothing(b) && isnothing(c) && isnothing(d))

# The matrix-side refresh behind `reinit!` and `cache.prob = prob`. A new state size or border
# width gets a new matrix, right-hand side and inner cache, built as at `init`.
function _refresh!(cache::DirectCache, refactor::Bool)
    prob = cache.prob
    if length(cache.rhs) != length(prob.f) + _border_width(prob.b)
        cache.matrix, cache.rhs, cache.inner = _direct_parts(prob, cache.alg, cache.kwargs)
    elseif refactor
        cache.inner.A = _assemble!(cache.matrix, prob)
    end
    return nothing
end

function SciMLBase.solve!(cache::DirectCache)
    cache.determinant_sign = nothing
    cache.stale && _sync!(cache, true)
    prob = cache.prob
    # Some backends solve in place and overwrite `b`, so it is restacked for every solve.
    cache.inner.b = _stack_rhs!(cache.rhs, prob)
    sol = solve!(cache.inner)
    if cache.alg.det_sign && SciMLBase.successful_retcode(sol.retcode)
        cache.determinant_sign = _lu_sign(cache.inner)
    end
    n, m = length(prob.f), _border_width(prob.b)
    # Give `v` the shape of the state back, so a 2D state comes out 2D.
    v = reshape(sol.u[1:n], size(prob.f))
    tail = sol.u[(n + 1):end]
    return BorderedLinearSolution(v, m == 1 ? tail[1] : tail, sol.retcode)
end
