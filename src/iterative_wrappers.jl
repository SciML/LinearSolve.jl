## Krylov.jl

struct KrylovJL{F,Tl,Tr,I,A,K} <: AbstractKrylovSubspaceMethod
    KrylovAlg::F
    Pl::Tl
    Pr::Tr
    gmres_restart::I
    window::I
    args::A
    kwargs::K
end

function KrylovJL(args...; KrylovAlg = Krylov.gmres!,
                  Pl=nothing, Pr=nothing,
                  gmres_restart=0, window=0,
                  kwargs...)

    Pl = (Pl === nothing) ? Identity() : Pl
    Pr = (Pr === nothing) ? Identity() : Pr

    return KrylovJL(KrylovAlg, Pl, Pr, gmres_restart, window,
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

function init_cacheval(alg::KrylovJL, A, b, u)

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
        solver = init_cacheval(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, solver)
    end

    M = get_preconditioner(alg.Pl, cache.Pl)
    N = get_preconditioner(alg.Pr, cache.Pr)

    M = (M === Identity()) ? I : inv(M)
    N = (N === Identity()) ? I : inv(N)

    atol    = cache.abstol
    rtol    = cache.reltol
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

    return SciMLBase.build_linear_solution(alg,cache.u,nothing)
end

## IterativeSolvers.jl

struct IterativeSolversJL{F,Tl,Tr,I,A,K} <: AbstractKrylovSubspaceMethod
    generate_iterator::F
    Pl::Tl
    Pr::Tr
    gmres_restart::I
    args::A
    kwargs::K
end

function IterativeSolversJL(args...;
                            generate_iterator = IterativeSolvers.gmres_iterable!,
                            Pl=nothing, Pr=nothing,
                            gmres_restart=0, kwargs...)

    Pl = (Pl === nothing) ? Identity() : Pl
    Pr = (Pr === nothing) ? Identity() : Pr

    return IterativeSolversJL(generate_iterator, Pl, Pr, gmres_restart,
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

function init_cacheval(alg::IterativeSolversJL, cache::LinearCache)
    @unpack A, b, u = cache

    Pl = get_preconditioner(alg.Pl, cache.Pl)
    Pr = get_preconditioner(alg.Pr, cache.Pr)

    abstol  = cache.abstol
    reltol  = cache.reltol
    maxiter = cache.maxiters
    verbose = cache.verbose

    restart = (alg.gmres_restart == 0) ? min(20, size(A,1)) : alg.gmres_restart

    kwargs = (abstol=abstol, reltol=reltol, maxiter=maxiter,
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
                              max_mv_products=maxiter*2,
                              alg.kwargs...)
    else # minres, qmr
        alg.generate_iterator(u, A, b, alg.args...;
                              abstol=abstol, reltol=reltol, maxiter=maxiter,
                              alg.kwargs...)
    end
    return iterable
end

function SciMLBase.solve(cache::LinearCache, alg::IterativeSolversJL; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache)
        cache = set_cacheval(cache, solver)
    end

    cache.verbose && println("Using IterativeSolvers.$(alg.generate_iterator)")
    for iter in enumerate(cache.cacheval)
        cache.verbose && println("Iter: $(iter[1]), residual: $(iter[2])")
        # TODO inject callbacks KSP into solve cb!(cache.cacheval)
    end
    cache.verbose && println()

    return SciMLBase.build_linear_solution(alg,cache.u,nothing)
end
