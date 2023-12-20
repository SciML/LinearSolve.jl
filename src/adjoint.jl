# TODO: Preconditioners? Should Pl be transposed and made Pr and similar for Pr.

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

function CRC.rrule(::typeof(SciMLBase.init), prob::LinearProblem,
        alg::SciMLLinearSolveAlgorithm, args...; kwargs...)
    cache = init(prob, alg, args...; kwargs...)
    function ∇init(∂cache)
        ∂∅ = NoTangent()
        ∂p = prob.p isa SciMLBase.NullParameters ? prob.p : ProjectTo(prob.p)(∂cache.p)
        ∂prob = LinearProblem(∂cache.A, ∂cache.b, ∂p)
        return (∂∅, ∂prob, ∂∅, ntuple(_ -> ∂∅, length(args))...)
    end
    return cache, ∇init
end

function CRC.rrule(::typeof(SciMLBase.solve!), cache::LinearCache, alg, args...;
        kwargs...)
    (; A, b, sensealg) = cache

    # Decide if we need to cache `A` and `b` for the reverse pass
    if sensealg.linsolve === nothing
        # We can reuse the factorization so no copy is needed
        # Krylov Methods don't modify `A`, so it's safe to just reuse it
        # No Copy is needed even for the default case
        if !(alg isa AbstractFactorization || alg isa AbstractKrylovSubspaceMethod ||
             alg isa DefaultLinearSolver)
            A_ = cache.alias_A ? deepcopy(A) : A
        end
    else
        error("Not Implemented Yet!!!")
    end

    # Forward Solve
    sol = solve!(cache, alg, args...; kwargs...)

    function ∇solve!(∂sol)
        @assert !cache.isfresh "`cache.A` has been updated between the forward and the \
                                reverse pass. This is not supported."
        ∂u = ∂sol.u
        if sensealg.linsolve === nothing
            λ = if cache.cacheval isa Factorization
                cache.cacheval' \ ∂u
            elseif cache.cacheval isa Tuple && cache.cacheval[1] isa Factorization
                first(cache.cacheval)' \ ∂u
            elseif alg isa AbstractKrylovSubspaceMethod
                invprob = LinearProblem(transpose(cache.A), ∂u)
                solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
            elseif alg isa DefaultLinearSolver
                LinearSolve.defaultalg_adjoint_eval(cache, ∂u)
            else
                invprob = LinearProblem(transpose(A_), ∂u) # We cached `A`
                solve(invprob, alg; cache.abstol, cache.reltol, cache.verbose).u
            end
        else
            error("Not Implemented Yet!!!")
        end

        ∂A = -λ * transpose(sol.u)
        ∂b = λ
        ∂∅ = NoTangent()

        ∂cache = LinearCache(∂A, ∂b, ∂∅, ∂∅, ∂∅, ∂∅, cache.isfresh, ∂∅, ∂∅, cache.abstol,
            cache.reltol, cache.maxiters, cache.verbose, cache.assumptions, cache.sensealg)

        return (∂∅, ∂cache, ∂∅, ntuple(_ -> ∂∅, length(args))...)
    end
    return sol, ∇solve!
end

function CRC.rrule(::Type{<:LinearProblem}, A, b, p; kwargs...)
    prob = LinearProblem(A, b, p)
    function ∇prob(∂prob)
        return NoTangent(), ∂prob.A, ∂prob.b, ∂prob.p
    end
    return prob, ∇prob
end
