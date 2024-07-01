module LinearSolveKrylovKitExt

using LinearSolve, KrylovKit, LinearAlgebra
using LinearSolve: LinearCache, DEFAULT_PRECS

function LinearSolve.KrylovKitJL(args...;
        KrylovAlg = KrylovKit.GMRES, gmres_restart = 0,
        precs = DEFAULT_PRECS,
        kwargs...)
    return KrylovKitJL(KrylovAlg, gmres_restart, precs, args, kwargs)
end

function LinearSolve.KrylovKitJL_CG(args...; kwargs...)
    KrylovKitJL(args...; KrylovAlg = KrylovKit.CG, kwargs..., isposdef = true)
end

function LinearSolve.KrylovKitJL_GMRES(args...; kwargs...)
    KrylovKitJL(args...; KrylovAlg = KrylovKit.GMRES, kwargs...)
end

LinearSolve.default_alias_A(::KrylovKitJL, ::Any, ::Any) = true
LinearSolve.default_alias_b(::KrylovKitJL, ::Any, ::Any) = true

function SciMLBase.solve!(cache::LinearCache, alg::KrylovKitJL; kwargs...)
    atol = float(cache.abstol)
    rtol = float(cache.reltol)
    maxiter = cache.maxiters
    verbosity = cache.verbose ? 1 : 0
    krylovdim = (alg.gmres_restart == 0) ? min(20, size(cache.A, 1)) : alg.gmres_restart

    kwargs = (atol = atol, rtol = rtol, maxiter = maxiter, verbosity = verbosity,
        krylovdim = krylovdim, alg.kwargs...)

    x, info = KrylovKit.linsolve(cache.A, cache.b, cache.u; kwargs...)

    copy!(cache.u, x)
    resid = info.normres
    retcode = info.converged == 1 ? ReturnCode.Default : ReturnCode.ConvergenceFailure
    iters = info.numiter
    return SciMLBase.build_linear_solution(alg, cache.u, resid, cache; retcode = retcode,
        iters = iters)
end

end
