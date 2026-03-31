# TODO: Preconditioners? Should Pl be transposed and made Pr and similar for Pr.

@doc doc"""
    LinearSolveAdjoint(; linsolve = missing)

Given a Linear Problem ``A x = b`` computes the sensitivities for ``A`` and ``b`` as:

```math
\begin{align}
A' \lambda &= \partial x   \\
\partial A  &= -\lambda x' \\
\partial b  &= \lambda
\end{align}
```

For more details, check [these notes](https://math.mit.edu/~stevenj/18.336/adjoint.pdf).

## Choice of Linear Solver

Note that in most cases, it makes sense to use the same linear solver for the adjoint as the
forward solve (this is done by keeping the linsolve as `missing`). For example, if the
forward solve was performed via a Factorization, then we can reuse the factorization for the
adjoint solve. However, for specific structured matrices if ``A'`` is known to have a
specific structure distinct from ``A`` then passing in a `linsolve` will be more efficient.
"""
@kwdef struct LinearSolveAdjoint{L} <:
    SciMLBase.AbstractSensitivityAlgorithm{0, false, :central}
    linsolve::L = missing
end
