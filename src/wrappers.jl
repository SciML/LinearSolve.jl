## Krylov.jl

struct KrylovJL{F,A,K} <: SciMLLinearSolveAlgorithm
    KrylovAlg::F
    args::A
    kwargs::K
end

function KrylovJL(args...; KrylovAlg = Krylov.gmres!, kwargs...)
    return KrylovJL(KrylovAlg, args, kwargs)
end

KrylovJL_CG(args...;kwargs...) = KrylovJL(Krylov.cg!, args...; kwargs...)
KrylovJL_GMRES(args...;kwargs...) = KrylovJL(Krylov.gmres!, args...; kwargs...)
KrylovJL_BICGSTAB(args...;kwargs...) = KrylovJL(Krylov.bicgstab!, args...; kwargs...)

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

## IterativeSolvers.jl

struct IterativeSolversJL{F,A,K} <: SciMLLinearSolveAlgorithm
    generate_iterator::F
    args::A
    kwargs::K
end

function IterativeSolversJL(args...;
                            generate_iterator = IterativeSolvers.gmres_iterable!,
                            kwargs...)
    return IterativeSolversJL(generate_iterator, args, kwargs)
end

#IterativeSolversJL_CG(args...; kwargs...)
#    = IterativeSolversJL(IterativeSolvers.cg_iterator!, args...; kwargs...)
#IterativeSolversJL_GMRES(args...;kwargs...)
#    = IterativeSolversJL(IterativeSolvers.gmres_iterable!, args...; kwargs...)
#IterativeSolversJL_BICGSTAB(args...;kwargs...)
#    = IterativeSolversJL(IterativeSolvers.bicgstabl_iterator!, args...;kwargs...)

function init_cacheval(alg::IterativeSolversJL, A, b, u)
    cacheval = if alg.generate_iterator === IterativeSolvers.cg_iterator!
        alg.generate_iterator(u, A, b)
    elseif alg.generate_iterator === IterativeSolvers.gmres_iterable!
        alg.generate_iterator(u, A, b)
    elseif alg.generate_iterator === IterativeSolvers.bicgstabl_iterator!
        alg.generate_iterator(u, A, b)
    else
        alg.generate_iterator(u, A, b)
    end
    return cacheval
end

function SciMLBase.solve(cache::LinearCache, alg::IterativeSolversJL; kwargs...)
    if cache.isfresh
        solver = init_cacheval(alg, cache.A, cache.b, cache.u)
        cache = set_cacheval(cache, solver)
    end

    for resi in cache.cacheval end

    return cache.u
end

