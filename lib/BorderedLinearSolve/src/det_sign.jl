"""
    det_sign(cache)

The sign of the determinant of the bordered matrix at the last `solve!`: `1` or `-1` for
a real matrix, `det / abs(det)` for a complex one, and `0` for a singular one whose solve
still succeeded.

Continuation uses it to keep its direction through a fold. There `det J` changes sign
but the bordered determinant does not. [`BorderingBLS`](@ref) multiplies the sign of
`det J` by that of the Schur complement `d - c'J⁻¹b`, which it already forms.
[`DirectBLS`](@ref) takes the sign of the assembled matrix.

The algorithm must be built with `det_sign = true` and `inner = LUFactorization()`. Each
successful `solve!` then reads the sign from the LU factors before they can change, with
`logabsdet`: one pass over the diagonal of a dense `U`, while a sparse LU copies its
factors out. A cache with no successful `solve!` since it was last built or updated
throws an `ArgumentError`, as does one whose algorithm keeps no sign.
"""
function det_sign(cache::Union{BorderedCache, DirectCache})
    cache.alg.det_sign || throw(
        ArgumentError(
            "det_sign needs an algorithm built with det_sign = true, " *
                "such as BorderingBLS(; inner = LUFactorization(), det_sign = true)"
        )
    )
    determinant_sign = cache.determinant_sign
    isnothing(determinant_sign) && throw(
        ArgumentError("det_sign needs a successful solve! since the cache was last built or updated")
    )
    return determinant_sign
end

# Only an LU factors the matrix itself. A normal-equation solver keeps factors of `A'A`,
# whose determinant is never negative.
function _check_signing(alg)
    !alg.det_sign || alg.inner isa LUFactorization || throw(
        ArgumentError(
            "det_sign = true needs inner = LUFactorization(), whose factors give the determinant, " *
                "but this algorithm has inner = $(alg.inner)"
        )
    )
    return nothing
end

_lu_sign(inner) = last(logabsdet(inner.cacheval))
