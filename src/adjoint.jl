# TODO: Preconditioners? Should Pl be transposed and made Pr and similar for Pr.
# TODO: Document the options in LinearSolveAdjoint

@doc doc"""
    LinearSolveAdjoint(; linsolve = nothing)

Given a Linear Problem ``A x = b`` computes the sensitivities for ``A`` and ``b`` as:

```math
\begin{align}
A^T \lambda &= \partial x   \\
\partial A  &= -\lambda x^T \\
\partial b  &= \lambda
\end{align}
```

For more details, check [these notes](https://math.mit.edu/~stevenj/18.336/adjoint.pdf).

## Choice of Linear Solver

Note that in most cases, it makes sense to use the same linear solver for the adjoint as the
forward solve (this is done by keeping the linsolve as `nothing`). For example, if the
forward solve was performed via a Factorization, then we can reuse the factorization for the
adjoint solve. However, for specific structured matrices if ``A^T`` is known to have a
specific structure distinct from ``A`` then passing in a `linsolve` will be more efficient.
"""
@kwdef struct LinearSolveAdjoint{L} <:
              SciMLBase.AbstractSensitivityAlgorithm{0, false, :central}
    linsolve::L = nothing
end

CRC.@non_differentiable SciMLBase.init(::LinearProblem, ::Any...)

function CRC.rrule(::typeof(SciMLBase.solve!), cache::LinearCache)
    sensealg = cache.sensealg

    # Decide if we need to cache the

    sol = solve!(cache)
    function ∇solve!(∂sol)
        @assert !cache.isfresh "`cache.A` has been updated between the forward and the reverse pass. This is not supported."

        ∂cache = NoTangent()
        return NoTangent(), ∂cache
    end
    return sol, ∇solve!
end
