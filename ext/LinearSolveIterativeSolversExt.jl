module LinearSolveIterativeSolversExt

using LinearSolve: LinearSolve, LinearCache, DEFAULT_PRECS, LinearVerbosity,
    OperatorAssumptions
import LinearSolve: IterativeSolversJL
using SciMLBase: SciMLBase
using LinearAlgebra: LinearAlgebra, norm
using SciMLLogging: SciMLLogging, @SciMLMessage

using IterativeSolvers: IterativeSolvers

function LinearSolve.IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.gmres_iterable!,
        gmres_restart = 0, precs = DEFAULT_PRECS, kwargs...
    )
    return IterativeSolversJL(
        generate_iterator, gmres_restart,
        precs, args, kwargs
    )
end

function LinearSolve.IterativeSolversJL_CG(args...; kwargs...)
    return IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.cg_iterator!,
        kwargs...
    )
end
function LinearSolve.IterativeSolversJL_GMRES(args...; kwargs...)
    return IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.gmres_iterable!,
        kwargs...
    )
end
function LinearSolve.IterativeSolversJL_IDRS(args...; kwargs...)
    return IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.idrs_iterable!,
        kwargs...
    )
end

function LinearSolve.IterativeSolversJL_BICGSTAB(args...; kwargs...)
    return IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.bicgstabl_iterator!,
        kwargs...
    )
end
function LinearSolve.IterativeSolversJL_MINRES(args...; kwargs...)
    return IterativeSolversJL(
        args...;
        generate_iterator = IterativeSolvers.minres_iterable!,
        kwargs...
    )
end

LinearSolve._isidentity_struct(::IterativeSolvers.Identity) = true

# Accept LinearSolve's `maxiters` spelling on the algorithm and hand
# IterativeSolvers the `maxiter` it expects. An explicit `maxiter` wins if both
# are given, and the NamedTuple is rebuilt rather than mutated so the result
# stays inferrable.
function _rename_maxiters(kwargs)
    haskey(kwargs, :maxiters) || return NamedTuple(kwargs)
    nt = NamedTuple(kwargs)
    maxiters = nt.maxiters
    rest = Base.structdiff(nt, NamedTuple{(:maxiters,)})
    return haskey(rest, :maxiter) ? rest : merge(rest, (; maxiter = maxiters))
end

function LinearSolve.init_cacheval(
        alg::IterativeSolversJL, A, b, u, Pl, Pr, maxiters::Int,
        abstol,
        reltol,
        verbose::Union{LinearVerbosity, Bool}, assumptions::OperatorAssumptions
    )
    if verbose isa Bool
        if verbose
            verbosity = LinearVerbosity(no_right_preconditioning = SciMLLogging.WarnLevel())
        else
            verbosity = LinearVerbosity(SciMLLogging.None())
        end
    else
        verbosity = verbose
    end
    restart = (alg.gmres_restart == 0) ? min(20, size(A, 1)) : alg.gmres_restart
    s = get(alg.kwargs, :idrs_s, 4) # shadow space

    # LinearSolve spells the iteration cap `maxiters`, IterativeSolvers spells it
    # `maxiter`. Passing the LinearSolve spelling on the algorithm, as in
    # `IterativeSolversJL_CG(maxiters = 100)`, used to forward an unknown keyword
    # and fail with a MethodError (SciML/LinearSolve.jl#175), so accept it as an
    # alias here. Everything below reads the cap from `maxiters_eff` so the
    # algorithm-level value also reaches the solvers that take it positionally.
    alg_kwargs = _rename_maxiters(alg.kwargs)
    maxiters_eff = get(alg_kwargs, :maxiter, maxiters)

    kwargs = (
        abstol = abstol, reltol = reltol, maxiter = maxiters_eff,
        alg_kwargs...,
    )

    iterable = if alg.generate_iterator === IterativeSolvers.cg_iterator!
        !LinearSolve._isidentity_struct(Pr) &&
            @SciMLMessage(
            "$(alg.generate_iterator) doesn't support right preconditioning",
            verbosity, :no_right_preconditioning
        )
        alg.generate_iterator(
            u, A, b, Pl;
            kwargs...
        )
    elseif alg.generate_iterator === IterativeSolvers.gmres_iterable!
        alg.generate_iterator(
            u, A, b; Pl = Pl, Pr = Pr, restart = restart,
            kwargs...
        )
    elseif alg.generate_iterator === IterativeSolvers.idrs_iterable!
        !!LinearSolve._isidentity_struct(Pr) &&
            @SciMLMessage(
            "$(alg.generate_iterator) doesn't support right preconditioning",
            verbosity, :no_right_preconditioning
        )
        history = IterativeSolvers.ConvergenceHistory(partial = true)
        history[:abstol] = abstol
        history[:reltol] = reltol
        # `idrs_iterable!` takes the tolerances and the iteration cap
        # positionally and accepts only `smoothing`/`verbose` as keywords, so
        # every name passed positionally below has to be dropped here. Filtering
        # `idrs_s` alone meant `IterativeSolversJL_IDRS(abstol = ...)` forwarded
        # `abstol` as a keyword too and failed with a `MethodError`
        # (SciML/LinearSolve.jl#24).
        function filter_kwargs(;
                idrs_s = 0, abstol = nothing, reltol = nothing,
                maxiter = nothing, kwargs...
            )
            return kwargs
        end
        IterativeSolvers.idrs_iterable!(
            history, u, A, b, s, Pl, abstol, reltol, maxiters_eff;
            filter_kwargs(; alg_kwargs...)...
        )
    elseif alg.generate_iterator === IterativeSolvers.bicgstabl_iterator!
        !!LinearSolve._isidentity_struct(Pr) &&
            @SciMLMessage(
            "$(alg.generate_iterator) doesn't support right preconditioning",
            verbosity, :no_right_preconditioning
        )
        # `bicgstabl_iterator!` caps work through `max_mv_products`, set just
        # above, and has no `maxiter` keyword at all, so the normalized cap has
        # to come out again here.
        alg.generate_iterator(
            u, A, b, alg.args...; Pl = Pl,
            abstol = abstol, reltol = reltol,
            max_mv_products = maxiters_eff * 2,
            Base.structdiff(alg_kwargs, NamedTuple{(:maxiter,)})...
        )
    else # minres, qmr
        alg.generate_iterator(
            u, A, b, alg.args...;
            abstol = abstol, reltol = reltol, maxiter = maxiters_eff,
            alg_kwargs...
        )
    end
    return iterable
end

function SciMLBase.solve!(cache::LinearCache, alg::IterativeSolversJL; kwargs...)
    if cache.precsisfresh && !isnothing(alg.precs)
        Pl, Pr = alg.precs(cache.Pl, cache.Pr)
        cache.Pl = Pl
        cache.Pr = Pr
        cache.precsisfresh = false
    end
    if cache.isfresh || !(cache.cacheval isa IterativeSolvers.GMRESIterable)
        solver = LinearSolve.init_cacheval(
            alg, cache.A, cache.b, cache.u, cache.Pl,
            cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol,
            cache.verbose,
            cache.assumptions
        )
        cache.cacheval = solver
        cache.isfresh = false
    end
    purge_history!(cache.cacheval, cache.u, cache.b)

    @SciMLMessage(
        "Using IterativeSolvers.$(alg.generate_iterator)",
        cache.verbose, :using_IterativeSolvers
    )
    i = 0
    for iter in enumerate(cache.cacheval)
        i += 1
        @SciMLMessage(
            "Iter: $(iter[1]), residual: $(iter[2])",
            cache.verbose, :IterativeSolvers_iterations
        )
        # TODO inject callbacks KSP into solve! cb!(cache.cacheval)
    end

    resid = _iterable_residual(cache.cacheval)
    if resid isa IterativeSolvers.Residual
        resid = resid.current
    end

    return SciMLBase.build_linear_solution(alg, cache.u, resid, nothing; iters = i)
end

# IterativeSolvers does not name this field consistently across its iterables.
# Reading `.residual` unconditionally made `IterativeSolversJL_MINRES` throw a
# `FieldError` on every solve, because `MINRESIterable` calls it `resnorm`
# (SciML/LinearSolve.jl#24).
_iterable_residual(iterable) = iterable.residual
_iterable_residual(iterable::IterativeSolvers.IDRSIterable) = iterable.R
_iterable_residual(iterable::IterativeSolvers.MINRESIterable) = iterable.resnorm

purge_history!(iter, x, b) = nothing
function purge_history!(iter::IterativeSolvers.GMRESIterable, x, b)
    iter.k = 1
    iter.x = x
    fill!(x, false)
    iter.b = b

    iter.residual.current = IterativeSolvers.init!(
        iter.arnoldi, iter.x, iter.b, iter.Pl,
        iter.Ax, initially_zero = true
    )
    IterativeSolvers.init_residual!(iter.residual, iter.residual.current)
    iter.β = iter.residual.current
    return nothing
end

# The constructors above all set the tolerance as follows.
#   tol = max(reltol * ||residual||, abstol)
#
# The iterable in turn is stored in `cache.cacheval`.
function update_tolerances_iterativesolversjl!(iter, atol, rtol)
    Rnorm = norm(iter.r)
    return iter.tol = max(rtol * Rnorm, atol)
end
function update_tolerances_iterativesolversjl!(iter::IterativeSolvers.GMRESIterable, atol, rtol)
    Rnorm = iter.residual.current
    return iter.tol = max(rtol * Rnorm, atol)
end
function update_tolerances_iterativesolversjl!(iter::IterativeSolvers.MINRESIterable, atol, rtol)
    Rnorm = norm(iter.v_curr)
    return iter.tol = max(rtol * Rnorm, atol)
end
function update_tolerances_iterativesolversjl!(iter::IterativeSolvers.IDRSIterable, atol, rtol)
    Rnorm = iter.normR
    return iter.tol = max(rtol * Rnorm, atol)
end

function LinearSolve.update_tolerances_internal!(cache, alg::IterativeSolversJL, atol, rtol)
    return update_tolerances_iterativesolversjl!(cache.cacheval, atol, rtol)
end

end
