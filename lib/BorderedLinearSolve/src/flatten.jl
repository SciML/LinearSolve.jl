# LinearSolve works with vector right-hand sides: a matrix `b` there means multiple
# right-hand sides, not a two-dimensional state. A state that is not already a vector
# is therefore flattened on the way into the inner solve and restored on the way out,
# so callers can keep a 2D state without writing `vec` and `reshape` themselves.
#
# This is a boundary adaptation, not a fix. The state keeps its own shape everywhere in
# the bordering algebra; only the inner solve sees the flattened form.

# A vector state needs none of this, and the operator is passed through untouched.
_needs_flattening(f) = !(f isa AbstractVector)

_flatten(x, f) = _needs_flattening(f) ? vec(x) : x
_unflatten(x, f) = _needs_flattening(f) ? reshape(x, size(f)) : x

"""
    _flatten_operator(J, f)

Present `J`, which acts on states shaped like `f`, as an operator on flattened vectors.

A `FunctionOperator` is used rather than a bespoke type because `LinearSolve` requires
an `AbstractSciMLOperator` for a matrix-free operator and rejects anything else.
"""
function _flatten_operator(J, f)
    _needs_flattening(f) || return J
    sz = size(f)
    proto = vec(similar(f))
    apply(v, u, p, t) = vec(J * reshape(v, sz))
    function apply(w, v, u, p, t)
        copyto!(w, vec(J * reshape(v, sz)))
        return w
    end
    return FunctionOperator(apply, proto, proto; islinear = true)
end
