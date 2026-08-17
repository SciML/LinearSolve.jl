"""
    LinearSolveAdjoint(; linsolve = missing, Pl = missing, Pr = missing)

Given a Linear Problem ``A x = b`` computes the sensitivities for ``A`` and ``b`` as:

```math
\\begin{align}
A' \\lambda &= \\partial x   \\\\
\\partial A  &= -\\lambda x' \\\\
\\partial b  &= \\lambda
\\end{align}
```

For more details, check [these notes](https://math.mit.edu/~stevenj/18.336/adjoint.pdf).

## Choice of Linear Solver

Note that in most cases, it makes sense to use the same linear solver for the adjoint as the
forward solve (this is done by keeping the linsolve as `missing`). For example, if the
forward solve was performed via a Factorization, then we can reuse the factorization for the
adjoint solve. However, for specific structured matrices if ``A'`` is known to have a
specific structure distinct from ``A`` then passing in a `linsolve` will be more efficient.

## Choice of Preconditioner

The adjoint solve is preconditioned from the forward preconditioners by default, which is
what keeps an iterative adjoint converging on a system that needs preconditioning at all.
Left preconditioning the forward system solves with the operator ``Pl^{-1} A``, whose
adjoint is ``A' Pl^{-*}``, that is the adjoint system *right* preconditioned by ``Pl^*``.
The two sides therefore swap and each is conjugated. Methods restricted to centered
preconditioning have no right slot to swap into, so for those the forward left
preconditioner is kept on the left; such methods apply to symmetric systems, where
``A' = A`` and the same preconditioner is the right one to reuse.

`Pl` and `Pr` override the respective sides of the adjoint solve. This is what a
matrix-free preconditioner needs, since it generally has no usable `adjoint`.
"""
@kwdef struct LinearSolveAdjoint{L, Tl, Tr} <:
    SciMLBase.AbstractSensitivityAlgorithm{0, false, :central}
    linsolve::L = missing
    Pl::Tl = missing
    Pr::Tr = missing
end

"""
    _adjoint_precs(alg, sensealg, Pl, Pr)

Preconditioner pair for the adjoint system ``A' λ = ∂x`` given the forward pair, following
the swap described in [`LinearSolveAdjoint`](@ref). Returns `nothing` for a side that
should fall back to the identity, matching the `Pl`/`Pr` keyword of `init`.
"""
function _adjoint_precs(alg, sensealg, Pl, Pr)
    # A sensealg other than `LinearSolveAdjoint` carries no overrides to honour.
    userPl = sensealg isa LinearSolveAdjoint ? sensealg.Pl : missing
    userPr = sensealg isa LinearSolveAdjoint ? sensealg.Pr : missing
    _side(user, derived) = user === missing ? _drop_identity(derived) : user
    # An algorithm carrying its own `precs` rebuilds the pair against `A'` when the adjoint
    # problem is initialized, so deriving one here would apply two preconditioners.
    if _has_own_precs(alg)
        return _side(userPl, nothing), _side(userPr, nothing)
    end
    if _supports_right_preconditioning(alg)
        return _side(userPl, adjoint(Pr)), _side(userPr, adjoint(Pl))
    end
    # No right slot to swap into: the forward left preconditioner stays on the left.
    return _side(userPl, adjoint(Pl)), _side(userPr, nothing)
end

_drop_identity(P) = (P === nothing || _isidentity_struct(P)) ? nothing : P

_has_own_precs(alg) = hasproperty(alg, :precs) && alg.precs !== nothing

# Mirrors the preconditioner dispatch in the `KrylovJL` `solve!`, where the methods for
# symmetric systems warn on and discard a right preconditioner.
_supports_right_preconditioning(_) = true
