module LinearSolveConjugateGradientsExt

using ConjugateGradients: ConjugateGradients, BiCGStabData, CGData, bicgstab!, cg!
using LinearAlgebra: LinearAlgebra
using LinearSolve: LinearSolve, ConjugateGradientsJL, LinearCache, LinearVerbosity,
    OperatorAssumptions
using SciMLBase: SciMLBase, ReturnCode
using SciMLLogging: @SciMLMessage

function LinearSolve.ConjugateGradientsJL_CG(; precs = nothing, kwargs...)
    return ConjugateGradientsJL(:cg, precs, kwargs)
end

function LinearSolve.ConjugateGradientsJL_BICGSTAB(; precs = nothing, kwargs...)
    return ConjugateGradientsJL(:bicgstab, precs, kwargs)
end

function ConjugateGradientsJL(; solver::Symbol = :cg, precs = nothing, kwargs...)
    return ConjugateGradientsJL(solver, precs, kwargs)
end

# ConjugateGradients.jl dispatches on `b::Vector{T} where {T <: Real}` and writes
# into an `x` of the same type, so anything else has no method at all. Checking at
# `init` keeps the failure at the point the algorithm was chosen, with a message
# naming the reason, rather than as a `MethodError` from inside the solver.
function _check_supported(alg::ConjugateGradientsJL, b, u)
    if !(b isa Vector && eltype(b) <: Real)
        throw(
            ArgumentError(
                "$(alg.solver === :cg ? "ConjugateGradientsJL_CG" : "ConjugateGradientsJL_BICGSTAB") " *
                    "requires a `Vector` right-hand side with a real element type, got " *
                    "$(typeof(b)). ConjugateGradients.jl defines its solvers only for " *
                    "`Vector{<:Real}`; use `KrylovJL_CG`/`KrylovJL_BICGSTAB` for complex or " *
                    "other array types."
            )
        )
    end
    if !(u isa Vector && eltype(u) === eltype(b))
        throw(
            ArgumentError(
                "ConjugateGradients.jl solves into a `Vector` matching the right-hand " *
                    "side's element type, got `u` of $(typeof(u)) for `b` of $(typeof(b))."
            )
        )
    end
    return nothing
end

# `CGData`/`BiCGStabData` hold the solver's working vectors, which is exactly what
# the cacheval is for: allocate once and reuse across solves.
function LinearSolve.init_cacheval(
        alg::ConjugateGradientsJL, A, b, u, Pl, Pr, maxiters::Int, abstol, reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    _check_supported(alg, b, u)
    T = eltype(b)
    n = length(b)
    return alg.solver === :cg ? CGData(n, T) : BiCGStabData(n, T)
end

# Positive exit codes are the converged ones, negative the failures; see
# `ConjugateGradients.reader`. Only `-2` means the iteration ran out of steps, so it
# is the one that maps to `MaxIters` rather than a plain failure.
function _retcode(exit_code::Integer)
    exit_code > 0 && return ReturnCode.Success
    exit_code == -2 && return ReturnCode.MaxIters
    return ReturnCode.Failure
end

function SciMLBase.solve!(cache::LinearCache, alg::ConjugateGradientsJL; kwargs...)
    if cache.precsisfresh && !isnothing(alg.precs)
        cache.Pl, cache.Pr = alg.precs(cache.A, cache.p)
        cache.precsisfresh = false
    end
    _check_supported(alg, cache.b, cache.u)

    # ConjugateGradients.jl takes a single left preconditioner as `precon`, applied
    # as `precon(z, r)`. `Pr` has nowhere to go, so refuse it rather than solve a
    # different problem than the caller asked for.
    if !LinearSolve._isidentity_struct(cache.Pr) && cache.Pr !== nothing
        throw(
            ArgumentError(
                "ConjugateGradientsJL supports a left preconditioner only; got a right " *
                    "preconditioner of $(typeof(cache.Pr))."
            )
        )
    end
    # `precon(z, r)` is called unconditionally, so the no-preconditioner case is
    # ConjugateGradients' own default of `copy!` rather than `nothing`.
    precon = if LinearSolve._isidentity_struct(cache.Pl) || cache.Pl === nothing
        copy!
    else
        (z, r) -> LinearAlgebra.ldiv!(z, cache.Pl, r)
    end

    data = LinearSolve.@get_cacheval(cache, :ConjugateGradientsJL)
    solver = alg.solver === :cg ? cg! : bicgstab!
    # ConjugateGradients.jl applies the operator as `A(output, input)` rather than
    # by multiplication, so the cache's matrix or operator is wrapped in that form.
    op = (y, x) -> LinearAlgebra.mul!(y, cache.A, x)
    exit_code, iters = solver(
        op, cache.b, cache.u;
        tol = float(cache.reltol), maxIter = cache.maxiters,
        precon, data, alg.kwargs...
    )

    retcode = _retcode(exit_code)
    if retcode !== ReturnCode.Success
        @SciMLMessage(
            "Solver failed: " * ConjugateGradients.reader(exit_code),
            cache.verbose, :solver_failure
        )
    end
    return SciMLBase.build_linear_solution(
        alg, cache.u, nothing, cache; retcode, iters
    )
end

end
