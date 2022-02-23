## Krylov.jl

struct KrylovJL{F,I,A,K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    window::I
    args::A
    kwargs::K
end

function KrylovJL(args...; KrylovAlg = Krylov.gmres!,
                  gmres_restart=0, window=0,
                  kwargs...)

    return KrylovJL(KrylovAlg, gmres_restart, window,
                    args, kwargs)
end

KrylovJL_CG(args...;kwargs...) =
    KrylovJL(args...; KrylovAlg=Krylov.cg!, kwargs...)
KrylovJL_GMRES(args...;kwargs...) =
    KrylovJL(args...; KrylovAlg=Krylov.gmres!, kwargs...)
KrylovJL_BICGSTAB(args...;kwargs...) =
    KrylovJL(args...; KrylovAlg=Krylov.bicgstab!, kwargs...)
KrylovJL_MINRES(args...;kwargs...) =
    KrylovJL(args...; KrylovAlg=Krylov.minres!, kwargs...)

function get_KrylovJL_solver(KrylovAlg)
    KS =
    if     (KrylovAlg === Krylov.lsmr!      ) Krylov.LsmrSolver
    elseif (KrylovAlg === Krylov.cgs!       ) Krylov.CgsSolver
    elseif (KrylovAlg === Krylov.usymlq!    ) Krylov.UsymlqSolver
    elseif (KrylovAlg === Krylov.lnlq!      ) Krylov.LnlqSolver
    elseif (KrylovAlg === Krylov.bicgstab!  ) Krylov.BicgstabSolver
    elseif (KrylovAlg === Krylov.crls!      ) Krylov.CrlsSolver
    elseif (KrylovAlg === Krylov.lsqr!      ) Krylov.LsqrSolver
    elseif (KrylovAlg === Krylov.minres!    ) Krylov.MinresSolver
    elseif (KrylovAlg === Krylov.cgne!      ) Krylov.CgneSolver
    elseif (KrylovAlg === Krylov.dqgmres!   ) Krylov.DqgmresSolver
    elseif (KrylovAlg === Krylov.symmlq!    ) Krylov.SymmlqSolver
    elseif (KrylovAlg === Krylov.trimr!     ) Krylov.TrimrSolver
    elseif (KrylovAlg === Krylov.usymqr!    ) Krylov.UsymqrSolver
    elseif (KrylovAlg === Krylov.bilqr!     ) Krylov.BilqrSolver
    elseif (KrylovAlg === Krylov.cr!        ) Krylov.CrSolver
    elseif (KrylovAlg === Krylov.craigmr!   ) Krylov.CraigmrSolver
    elseif (KrylovAlg === Krylov.tricg!     ) Krylov.TricgSolver
    elseif (KrylovAlg === Krylov.craig!     ) Krylov.CraigSolver
    elseif (KrylovAlg === Krylov.diom!      ) Krylov.DiomSolver
    elseif (KrylovAlg === Krylov.lslq!      ) Krylov.LslqSolver
    elseif (KrylovAlg === Krylov.trilqr!    ) Krylov.TrilqrSolver
    elseif (KrylovAlg === Krylov.crmr!      ) Krylov.CrmrSolver
    elseif (KrylovAlg === Krylov.cg!        ) Krylov.CgSolver
    elseif (KrylovAlg === Krylov.cg_lanczos!) Krylov.CgLanczosShiftSolver
    elseif (KrylovAlg === Krylov.cgls!      ) Krylov.CglsSolver
    elseif (KrylovAlg === Krylov.cg_lanczos!) Krylov.CgLanczosSolver
    elseif (KrylovAlg === Krylov.bilq!      ) Krylov.BilqSolver
    elseif (KrylovAlg === Krylov.minres_qlp!) Krylov.MinresQlpSolver
    elseif (KrylovAlg === Krylov.qmr!       ) Krylov.QmrSolver
    elseif (KrylovAlg === Krylov.gmres!     ) Krylov.GmresSolver
    elseif (KrylovAlg === Krylov.fom!       ) Krylov.FomSolver
    end

    return KS
end

function init_cacheval(alg::KrylovJL, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)

    KS = get_KrylovJL_solver(alg.KrylovAlg)

    memory = (alg.gmres_restart == 0) ? min(20, size(A,1)) : alg.gmres_restart

    solver = if(
        alg.KrylovAlg === Krylov.dqgmres! ||
        alg.KrylovAlg === Krylov.diom!    ||
        alg.KrylovAlg === Krylov.gmres!   ||
        alg.KrylovAlg === Krylov.fom!
       )
        KS(A, b, memory)
    elseif(
           alg.KrylovAlg === Krylov.minres! ||
           alg.KrylovAlg === Krylov.symmlq! ||
           alg.KrylovAlg === Krylov.lslq!   ||
           alg.KrylovAlg === Krylov.lsqr!   ||
           alg.KrylovAlg === Krylov.lsmr!
          )
        (alg.window != 0) ? KS(A,b; window=alg.window) : KS(A, b)
    else
        KS(A, b)
    end

    solver.x = u

    return solver
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovJL; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr, cache.maxiters, cache.abstol, cache.reltol, cache.verbose)
        cache = set_cacheval(cache, solver)
    end

    M = cache.Pl
    N = cache.Pr

    M = (M === Identity()) ? I : InvPreconditioner(M)
    N = (N === Identity()) ? I : InvPreconditioner(N)

    atol    = float(cache.abstol)
    rtol    = float(cache.reltol)
    itmax   = cache.maxiters
    verbose = cache.verbose ? 1 : 0

    args   = (cache.cacheval, cache.A, cache.b)
    kwargs = (atol=atol, rtol=rtol, itmax=itmax, verbose=verbose,
              alg.kwargs...)

    if cache.cacheval isa Krylov.CgSolver
        N !== I  &&
            @warn "$(alg.KrylovAlg) doesn't support right preconditioning."
        Krylov.solve!(args...; M=M,
                      kwargs...)
    elseif cache.cacheval isa Krylov.GmresSolver
        Krylov.solve!(args...; M=M, N=N,
                      kwargs...)
    elseif cache.cacheval isa Krylov.BicgstabSolver
        Krylov.solve!(args...; M=M, N=N,
                      kwargs...)
    elseif cache.cacheval isa Krylov.MinresSolver
        N !== I  &&
            @warn "$(alg.KrylovAlg) doesn't support right preconditioning."
        Krylov.solve!(args...; M=M,
                      kwargs...)
    else
        Krylov.solve!(args...; kwargs...)
    end

    return SciMLBase.build_linear_solution(alg,cache.u,Krylov.Aprod(cache.cacheval),cache)
end

## IterativeSolvers.jl

struct IterativeSolversJL{F,I,A,K} <: AbstractKrylovSubspaceMethod
    generate_iterator::F
    gmres_restart::I
    args::A
    kwargs::K
end

function IterativeSolversJL(args...;
                            generate_iterator = IterativeSolvers.gmres_iterable!,
                            gmres_restart=0, kwargs...)

    return IterativeSolversJL(generate_iterator, gmres_restart,
                              args, kwargs)
end

IterativeSolversJL_CG(args...; kwargs...) =
    IterativeSolversJL(args...;
                       generate_iterator=IterativeSolvers.cg_iterator!,
                       kwargs...)
IterativeSolversJL_GMRES(args...;kwargs...) =
    IterativeSolversJL(args...;
                       generate_iterator=IterativeSolvers.gmres_iterable!,
                       kwargs...)
IterativeSolversJL_BICGSTAB(args...;kwargs...) =
    IterativeSolversJL(args...;
                       generate_iterator=IterativeSolvers.bicgstabl_iterator!,
                       kwargs...)
IterativeSolversJL_MINRES(args...;kwargs...) =
    IterativeSolversJL(args...;
                       generate_iterator=IterativeSolvers.minres_iterable!,
                       kwargs...)

function init_cacheval(alg::IterativeSolversJL, A, b, u, Pl, Pr, maxiters, abstol, reltol, verbose)
    restart = (alg.gmres_restart == 0) ? min(20, size(A,1)) : alg.gmres_restart

    kwargs = (abstol=abstol, reltol=reltol, maxiter=maxiters,
              alg.kwargs...)

    iterable = if alg.generate_iterator === IterativeSolvers.cg_iterator!
        Pr !== Identity() &&
          @warn "$(alg.generate_iterator) doesn't support right preconditioning"
        alg.generate_iterator(u, A, b, Pl;
                              kwargs...)
    elseif alg.generate_iterator === IterativeSolvers.gmres_iterable!
        alg.generate_iterator(u, A, b; Pl=Pl, Pr=Pr, restart=restart,
                              kwargs...)
    elseif alg.generate_iterator === IterativeSolvers.bicgstabl_iterator!
        Pr !== Identity() &&
          @warn "$(alg.generate_iterator) doesn't support right preconditioning"
        alg.generate_iterator(u, A, b, alg.args...; Pl=Pl,
                              abstol=abstol, reltol=reltol,
                              max_mv_products=maxiters*2,
                              alg.kwargs...)
    else # minres, qmr
        alg.generate_iterator(u, A, b, alg.args...;
                              abstol=abstol, reltol=reltol, maxiter=maxiters,
                              alg.kwargs...)
    end
    return iterable
end

function SciMLBase.solve(cache::LinearCache, alg::IterativeSolversJL; kwargs...)
    if cache.isfresh || !(typeof(alg) <: IterativeSolvers.GMRESIterable)
        solver = init_cacheval(alg, cache.A, cache.b, cache.u, cache.Pl, cache.Pr, cache.maxiters, cache.abstol, cache.reltol, cache.verbose)
        cache = set_cacheval(cache, solver)
    end
    purge_history!(cache.cacheval, cache.u, cache.b)

    cache.verbose && println("Using IterativeSolvers.$(alg.generate_iterator)")
    i = 0
    for iter in enumerate(cache.cacheval)
        i += 1
        cache.verbose && println("Iter: $(iter[1]), residual: $(iter[2])")
        # TODO inject callbacks KSP into solve cb!(cache.cacheval)
    end
    cache.verbose && println()

    return SciMLBase.build_linear_solution(alg,cache.u,nothing,cache; iters = i)
end

purge_history!(iter, x, b) = nothing
function purge_history!(iter::IterativeSolvers.GMRESIterable, x, b)
  iter.k = 1
  iter.x  = x
  fill!(x,false)
  iter.b  = b

  iter.residual.current = IterativeSolvers.init!(iter.arnoldi, iter.x, iter.b, iter.Pl, iter.Ax, initially_zero = true)
  IterativeSolvers.init_residual!(iter.residual, iter.residual.current)
  iter.β = iter.residual.current
  nothing
end

## KrylovKit.jl

struct KrylovKitJL{F,I,K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    gmres_restart::I
    kwargs::K
end

function KrylovKitJL(KrylovAlg = KrylovKit.GMRES, gmres_restart=0, kwargs...)
    return KrylovJL(KrylovAlg, gmres_restart, kwargs)
end

KrylovKitJL_GMRES(kwargs...) = KrylovKitJL()
KrylovKitJL_CG(kwargs...) = KrylovKitJL(KrylovKitAlg=KrylovKit.CG, kwargs...)

function SciMLBase.solve(cache::LinearCache, alg::KrylovKitJL, kwargs...)

    atol      = float(cache.abstol)
    rtol      = float(cache.reltol)
    maxiter   = cache.maxiters
    verbosity = cache.verbose ? 1 : 0
    krylovdim = alg.gmres_restart

    kwargs = (atol=atol, rtol=rtol, maxiter=maxiter, verbosity=verbosity,
              krylovdim = krylovdim, alg.kwargs...)

    x, info = KrylovKit.linsolve(cache.A, cache.b, cache.u, alg.KrylovAlg,
                                 [a₀::Number = 0, a₁::Number = 1])
    copy!(cache.u, x)
    resid = info.normres
    retcode = info.converged == 1 ? :Default : :DidNotConverge
    iters = info.numiter

    return SciMLBase.build_linear_solution(alg,cache.u, resid,cache;
                                           retcode = retcode, iters = iters)
end
