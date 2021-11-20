## Krylov.jl

struct KrylovJL{F,A,K} <: SciMLLinearSolveAlgorithm
    KrylovAlg::F
    args::A
    kwargs::K
end

function KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
    return KrylovJL(KrylovAlg, args, kwargs)
end

function init_cacheval(alg::KrylovJL, A, b, u)
    cacheval = if alg.KrylovAlg === Krylov.cg!
        Krylov.CgSolver(A,b)
    elseif alg.KrylovAlg === Krylov.gmres!
        Krylov.GmresSolver(A,b,20)
    elseif alg.KrylovAlg === Krylov.bicgstab!
        Krylov.BicgstabSolver(A,b)
    else
        nothing
    end
    return cacheval
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovJL; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, solver)
    end

    cache.cacheval.x = cache.u
    alg.KrylovAlg(cache.cacheval, cache.A, cache.b;
                  M=cache.Pl, N=cache.Pr, alg.kwargs...)

    return cache.u
end

KrylovJL_CG(args...;kwargs...) = KrylovJL(Krylov.cg!, args...; kwargs...)
KrylovJL_GMRES(args...;kwargs...) = KrylovJL(Krylov.gmres!, args...; kwargs...)
KrylovJL_BICGSTAB(args...;kwargs...) = KrylovJL(Krylov.bicgstab!, args...; kwargs...)

## IterativeSolvers.jl

struct IterativeSolversJL{F,A,K} <: SciMLLinearSolveAlgorithm
    solver::F
    args::A
    kwargs::K
end

## KrylovKit.jl

struct KrylovKitJL{F,A,K} <: SciMLLinearSolveAlgorithm
    solver::F
    args::A
    kwargs::K
end

function KrylovKitJL(args...; solver = KrylovKit.CG(), kwargs...)
    return KrylovKitJL(solver, args, kwargs)
end

function SciMLBase.solve(cache::LinearCache, alg::KrylovKitJL,args...;kwargs...)
    @unpack A, b, u = cache
    @unpack solver = alg
    u = KrylovKit.linsolve(A, b, u, solver, args...; kwargs...)[1] #no precond?!
    return u
end

