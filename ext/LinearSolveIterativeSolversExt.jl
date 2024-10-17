module LinearSolveIterativeSolversExt

using LinearSolve, LinearAlgebra
using LinearSolve: LinearCache, DEFAULT_PRECS
import LinearSolve: IterativeSolversJL

if isdefined(Base, :get_extension)
    using IterativeSolvers
else
    using ..IterativeSolvers
end

function LinearSolve.IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.gmres_iterable!,
        gmres_restart = 0, precs = DEFAULT_PRECS, kwargs...)
    return IterativeSolversJL(generate_iterator, gmres_restart,
        precs, args, kwargs)
end

function LinearSolve.IterativeSolversJL_CG(args...; kwargs...)
    IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.cg_iterator!,
        kwargs...)
end
function LinearSolve.IterativeSolversJL_GMRES(args...; kwargs...)
    IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.gmres_iterable!,
        kwargs...)
end
function LinearSolve.IterativeSolversJL_IDRS(args...; kwargs...)
    IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.idrs_iterable!,
        kwargs...)
end

function LinearSolve.IterativeSolversJL_BICGSTAB(args...; kwargs...)
    IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.bicgstabl_iterator!,
        kwargs...)
end
function LinearSolve.IterativeSolversJL_MINRES(args...; kwargs...)
    IterativeSolversJL(args...;
        generate_iterator = IterativeSolvers.minres_iterable!,
        kwargs...)
end

LinearSolve._isidentity_struct(::IterativeSolvers.Identity) = true
LinearSolve.default_alias_A(::IterativeSolversJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::IterativeSolversJL, ::Any, ::Any) = true

function LinearSolve.init_cacheval(alg::IterativeSolversJL, A, b, u, Pl, Pr, maxiters::Int,
        abstol,
        reltol,
        verbose::Bool, assumptions::OperatorAssumptions)
    restart = (alg.gmres_restart == 0) ? min(20, size(A, 1)) : alg.gmres_restart
    s = :idrs_s in keys(alg.kwargs) ? alg.kwargs.idrs_s : 4 # shadow space

    kwargs = (abstol = abstol, reltol = reltol, maxiter = maxiters,
        alg.kwargs...)

    iterable = if alg.generate_iterator === IterativeSolvers.cg_iterator!
        !LinearSolve._isidentity_struct(Pr) &&
            @warn "$(alg.generate_iterator) doesn't support right preconditioning"
        alg.generate_iterator(u, A, b, Pl;
            kwargs...)
    elseif alg.generate_iterator === IterativeSolvers.gmres_iterable!
        alg.generate_iterator(u, A, b; Pl = Pl, Pr = Pr, restart = restart,
            kwargs...)
    elseif alg.generate_iterator === IterativeSolvers.idrs_iterable!
        !!LinearSolve._isidentity_struct(Pr) &&
            @warn "$(alg.generate_iterator) doesn't support right preconditioning"
        history = IterativeSolvers.ConvergenceHistory(partial = true)
        history[:abstol] = abstol
        history[:reltol] = reltol
        IterativeSolvers.idrs_iterable!(history, u, A, b, s, Pl, abstol, reltol, maxiters;
            alg.kwargs...)
    elseif alg.generate_iterator === IterativeSolvers.bicgstabl_iterator!
        !!LinearSolve._isidentity_struct(Pr) &&
            @warn "$(alg.generate_iterator) doesn't support right preconditioning"
        alg.generate_iterator(u, A, b, alg.args...; Pl = Pl,
            abstol = abstol, reltol = reltol,
            max_mv_products = maxiters * 2,
            alg.kwargs...)
    else # minres, qmr
        alg.generate_iterator(u, A, b, alg.args...;
            abstol = abstol, reltol = reltol, maxiter = maxiters,
            alg.kwargs...)
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
    if cache.isfresh || !(alg isa IterativeSolvers.GMRESIterable)
        solver = LinearSolve.init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl,
            cache.Pr,
            cache.maxiters, cache.abstol, cache.reltol,
            cache.verbose,
            cache.assumptions)
        cache.cacheval = solver
        cache.isfresh = false
    end
    purge_history!(cache.cacheval, cache.u, cache.b)

    cache.verbose && println("Using IterativeSolvers.$(alg.generate_iterator)")
    i = 0
    for iter in enumerate(cache.cacheval)
        i += 1
        cache.verbose && println("Iter: $(iter[1]), residual: $(iter[2])")
        # TODO inject callbacks KSP into solve! cb!(cache.cacheval)
    end
    cache.verbose && println()

    resid = cache.cacheval isa IterativeSolvers.IDRSIterable ? cache.cacheval.R :
            cache.cacheval.residual
    if resid isa IterativeSolvers.Residual
        resid = resid.current
    end

    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache; iters = i)
end

purge_history!(iter, x, b) = nothing
function purge_history!(iter::IterativeSolvers.GMRESIterable, x, b)
    iter.k = 1
    iter.x = x
    fill!(x, false)
    iter.b = b

    iter.residual.current = IterativeSolvers.init!(iter.arnoldi, iter.x, iter.b, iter.Pl,
        iter.Ax, initially_zero = true)
    IterativeSolvers.init_residual!(iter.residual, iter.residual.current)
    iter.β = iter.residual.current
    nothing
end

end
